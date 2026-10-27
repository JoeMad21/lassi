# Spike: performance metrics on the Tenstorrent cards (wall time, cycles, FLOPs), and what ttsim gives

- Owner request (2026-10-26, the owner's words, from the session brief): "I want to track performance metrics on the
  Tenstorrent (wall clock time, flops, etc.)."
- Bible: Agent Rules 1, 2, 9, 10, 14; Component Interfaces (the Profiler row and the Profiler contract bullet);
  Result Record (Attempt.run, Attempt.profile, Profiles, Simulator wall times, Storage); Project Recipes, Notes (the
  Profilers bullet); Execution Backends (the ttsim and tt_silicon rows; ttsim Facts); Toolchain Pins (Install);
  Training Module (Compute, Safeguards); Evaluation Protocol (Reporting Rules); Environment State.
- Related: plans/spikes/p9-tt-power.md (OQ-093, the sysfs power source; on branch p9-ee), plans/spikes/tt-training.md
  (OQ-100, OQ-101, OQ-102), plans/spikes/tt-silicon-executor.md, plans/spikes/p4-ttsim-runtime.md,
  results/tt-silicon-smoke-20261023, the uncommitted backend lassi/profilers/tenstorrent.py in c:/dev/lassi (branch
  p9-ee, task P9.5; read, not changed).
- Status: sources only. No rx command of any kind ran, nothing touched alpha01, and no device command ran anywhere.
  Upstream files were read from GitHub (raw files and the git trees API) on the workstation (Windows, Git Bash) on
  2026-10-26. Two members of one published wheel were fetched by HTTP range requests and inflated into a session
  scratch directory, then searched for strings; nothing from them was executed. Nothing here is [MEASURED]; every
  number not read from a source is PROJECTED arithmetic, and every finding says whether it was read, inferred, or is
  unknown.
- Date: 2026-10-26.
- Pins (toolchains/tt-metal.pin, toolchains/ttsim.pin, toolchains/tt-mlir.pin): tt-metal
  5280a9cfb00998fd49667a29523d03aee905c129, built with `-DENABLE_TRACY=OFF` among its CMAKE_FLAGS and only the
  targets tt_metal and hw_toolchain; its submodules UMD 0450f1be3a21cbf17adb97517038586a6a3af4c0 and Tracy
  0aaefbb689b4c60694edc905545fc4709fd13f6a (the pin's SUBMODULES line); ttsim v1.3.4 (tag commit
  2174bdb9dce5acbd775fad3fbfb96710ffe1bd87); tt-mlir 5f396fd6ef85780bf1b6b85bef9c7d2f84a85f3c. The driver on alpha01
  is tenstorrent 2.8.0, read as tt-kmd tag ttkmd-2.8.0 (commit 0b8b5e7c0c44436740ca9ec545e2d08a5c2dfdf1), as the
  sibling spikes assume.
- Citations:
  - `metal <file>:<line>`: tenstorrent/tt-metal at 5280a9cf.
  - `umd <file>:<line>`: tenstorrent/tt-umd at 0450f1be, paths under device/.
  - `kmd <file>:<line>`: tenstorrent/tt-kmd at ttkmd-2.8.0.
  - `tracy <file>:<line>`: tenstorrent/tracy at 0aaefbb6.
  - `mlir <file>:<line>`: tenstorrent/tt-mlir at 5f396fd6.
  - `xla <file>:<line>`: tenstorrent/tt-xla at ff3a67dde32ad3311258674bd3fa260e11a20e98, the commit the wheel
    pjrt-plugin-tt 0.9.0.dev20260223001148 names (plans/spikes/tt-training.md).
  - `ttsim README:<line>`: tenstorrent/ttsim at v1.3.4.
  - `wheel <member>`: a member of pjrt_plugin_tt-0.9.0.dev20260223001148-cp311-cp311-linux_x86_64.whl.
  - `bible <line>`: docs/BIBLE.md in this worktree (mirror rev 435); `bible-p9 <line>`: docs/BIBLE.md on branch p9-ee
    (mirror rev 395), which holds P9.1's energy profiler.
- Labels: [read] means read in the cited source; [inferred] means reasoned from what was read, not seen directly;
  [unknown] means no source read here settles it.
- Conventions: upstream code is paraphrased, never quoted, except short identifiers, file names, CSV headers, and
  log strings, which are reproduced exactly because the design parses them.

## Question

What performance metrics can LASSI record for programs on alpha01's Wormhole n300 cards at the pins (wall-clock
time, device time, cycles, clocks, FLOPs, power, temperature), how, at what cost, and where (which process, which
files)? What does ttsim report, and how may LASSI record it without calling it performance? How are FLOPs counted
(analytic per item, or hardware counters)? What changes for the training runs on the cards (step time, tokens per
second, achieved FLOP/s)? How does all of it map onto the bible's Profiler interface, Result Record, and Reporting
Rules, and what needs the owner?

Why it matters:

- The owner asked for it by name. The silicon runs in view are tracked reference programs a human starts (Agent
  Rule 9; the tt_silicon executor's refusals, lassi/executors/tt_silicon.py, unchanged by this spike) and training
  on the cards (task P17.38; OQ-100 answered (a)). Model output on silicon still waits on OQ-078 and OQ-099.
- Tasks that depend on it: the tt_silicon executor's follow-ups (a device-timing reading), P9.5 and P9.6 (backend
  `tt`, which already samples sysfs power and could sample clocks and temperature the same way), P17.38 (what a
  training run on a card records per step), lassi.analysis (any derived FLOP/s), and the Reporting Rules.
- Agent Rule 2 forbids reporting simulator timing as performance, and Agent Rule 1 forbids relabeling or
  extrapolating a measurement, so the design must say per value which device, which clock, and which denominator.

## Classification

- What the device profiler, Tracy, Inspector, UMD, the driver's sysfs files, and ttsim expose at the pins, and what
  the pinned install can and cannot do: factual, answered by reading the pinned sources (Findings 1 to 9).
- Whether the tt-xla training wheel carries a Tracy-enabled tt-metal: factual, answered by reading the wheel's file
  list and two of its libraries' strings (Finding 10).
- The values on alpha01's cards (AICLK under load, the telemetry refresh interval, profiler overhead on these cards,
  whether a Tracy-enabled build runs in the sandbox): factual but unknown without runs on the cards, which need the
  owner's approval (Agent Rule 17) and are listed under Unknown.
- How LASSI gets device kernel time (a second, Tracy-enabled tt-metal install, a change of the one pin, or none): a
  choice, since it is a pin change and a build on alpha01 (OQ-104 draft).
- Which FLOP convention and which peak LASSI reports: a choice of reporting convention (OQ-105 draft).
- Whether a training process that loads the wheel's Tracy client may run on alpha01's network: a policy choice for
  P17.38 (OQ-106 draft).

## Plain answer

- At the pins, the pinned install cannot profile the device. tt-metal's device profiler (per-core, per-RISC cycle
  stamps, kernel and firmware durations, the program-level perf APIs, and the FPU utilization counters) is compiled
  only when tt-metal is built with ENABLE_TRACY=ON, and LASSI's pin builds it OFF; setting TT_METAL_DEVICE_PROFILER=1
  against the pinned build aborts the program at start-up [read: metal tt_metal/llrt/rtoptions.cpp:673-681;
  toolchains/tt-metal.pin CMAKE_FLAGS].
- What the cards give at the pins without any build change [read for the mechanism; values unknown]: the whole
  process's wall time (already in Attempt.run.wall_s and the timing profiler's runtime_s), host-side JIT compile
  durations from tt-metal's Inspector log, and the driver's sysfs telemetry, which a sampler reads without opening
  the card: AICLK, AXICLK, and ARCCLK in MHz, the PCIe chip's core power, ASIC temperature, core voltage, and
  current. Process wall time is about 1.8 to 1.9 s per tracked reference [MEASURED 2026-10-23,
  results/tt-silicon-smoke-20261023], dominated by device open, JIT, and close, while a 640 x 640 x 640 matmul at the
  vendor's peak would take about 8 microseconds on one chip (PROJECTED), so process wall time says almost nothing
  about kernel speed.
- With a Tracy-enabled build of the same commit, the device profiler records, per program, the device kernel
  duration in cycles and in nanoseconds (cycles converted at the AICLK the card reports through an ARC message), per
  core and per RISC, into profile_log_device.csv and cpp_device_perf_report.csv, read back at mesh device close; all
  five tracked references close their mesh device, so they need no code change [read; inferred for the
  references].
- ttsim gives no device time. Its clock advances only when the host reads from it (UMD advances it 10 cycles per
  read), UMD reports its AICLK as 0, and the README lists cycle counters and timers among the reads that may diverge
  from silicon [read]. LASSI records only the simulator wall time it already labels "not performance" and keeps the
  `[<cycle>]` stamp of a finding line inside the finding's text; it computes no rate from either.
- FLOPs come from the items, not the device: Wormhole's FPU counters, which the pinned profiler can read in a
  Tracy-enabled build, give busy-cycle utilization, not floating-point operation counts [read]. Analytic counts from
  tt-pairs-v0's shapes: matmul_single_core and matmul_multi_core 2 x 640^3 = 524,288,000 FLOP each; eltwise_binary
  65,536 FLOP; eltwise_sfpu 65,536 exp evaluations; loopback 0 FLOP and 204,800 bytes moved.
- Training: step time and tokens per second are measured by the trainer on the host around a device sync; achieved
  FLOP/s is analytic model FLOPs per step (PaLM's 6N + 12LHQT for full weights) over the measured step time. The
  tt-xla wheel that OQ-101 (a) would train with carries a Tracy-enabled tt-metal [read from the wheel's strings], so
  its device profiler is available without a build, but its Tracy client would also listen on TCP 8086 and broadcast
  on UDP 255.255.255.255:8086 every 3 s from library load, on alpha01's network unless the process runs in its own
  network namespace [inferred from the Tracy source and the wheel's strings; OQ-106 draft].

## Commands and outputs (trimmed)

Every command ran on the workstation, into a session scratch directory under the user's temporary directory
(downloads in one subdirectory, and in another the two helper scripts zipnames.py and zipmember.py, which read zip
bytes only). Python ran as `uv run --no-project --isolated python -I`.

```
gh api "repos/tenstorrent/tt-metal/git/trees/5280a9cfb00998fd49667a29523d03aee905c129?recursive=1"
  truncated: false, 17701 entries; profiler files listed include tt_metal/impl/profiler/{profiler.cpp 132777 B,
  tt_metal_profiler.cpp 57096 B, profiler_analysis.cpp 33964 B, profiler_state_manager.cpp, profiler_paths.hpp},
  tt_metal/tools/profiler/{kernel_profiler.hpp, perf_counters.hpp, cpp_device_analyses.json},
  tt_metal/api/tt-metalium/{tt_metal_profiler.hpp, experimental/profiler.hpp}, tools/tracy/*.py,
  docs/source/tt-metalium/tools/{device_program_profiler.rst, tracy_profiler.rst}

curl -sSfL https://raw.githubusercontent.com/tenstorrent/tt-metal/5280a9cf.../<file>   (37 files; one 404:
  tt_metal/ttsim-version, which postdates the pin, as the bible's ttsim Facts say)
curl -sSfL https://raw.githubusercontent.com/tenstorrent/tt-umd/0450f1be.../device/<file>   (21 files)
curl -sSfL https://raw.githubusercontent.com/tenstorrent/tt-kmd/0b8b5e7c.../{wormhole.c,telemetry.h,hwmon.c}
curl -sSfL https://raw.githubusercontent.com/tenstorrent/tracy/0aaefbb6.../{CMakeLists.txt,
  public/client/TracyProfiler.cpp, public/common/TracySocket.cpp}
curl -sSfL https://raw.githubusercontent.com/tenstorrent/tt-mlir/5f396fd6.../{CMakeLists.txt,third_party/CMakeLists.txt}
curl -sSfL https://raw.githubusercontent.com/tenstorrent/tt-xla/ff3a67dd.../{CMakeLists.txt,third_party/CMakeLists.txt,
  python_package/setup.py,.github/workflows/call-build.yml}
gh api repos/tenstorrent/ttsim/git/refs/tags/v1.3.4      -> commit 2174bdb9dce5acbd775fad3fbfb96710ffe1bd87
gh api "repos/tenstorrent/ttsim/git/trees/v1.3.4?recursive=1"   -> README.md and licence files only (binary releases)

grep -n "TRACY_ENABLE\|#if\|#endif" tt_metal/impl/profiler/tt_metal_profiler.cpp
  64,78,592,707,714,803,878,896,947,1010,1054,1070,1112,1251,1283,1337: every device-profiler host function body
  sits inside #if defined(TRACY_ENABLE)
grep -n "SimulationChip::get_clock\|TTSimTTDevice::get_clock\|libttsim_clock" umd/device/...
  simulation/simulation_chip.cpp:110: int SimulationChip::get_clock() { return 0; }
  tt_device/tt_sim_tt_device.cpp:120: pfn_libttsim_clock(10);   (the only call, after each tile read)
  tt_device/tt_sim_tt_device.cpp:230-232: get_clock throws "not supported in TTSim simulation device"

curl -sSL https://pypi.eng.aws.tenstorrent.com/pjrt-plugin-tt/ | grep 0.9.0.dev20260223001148
  pjrt_plugin_tt-0.9.0.dev20260223001148-cp311-cp311-linux_x86_64.whl#sha256=fbbe6241869e9646...b0c4db
curl -sSI <wheel URL>         Content-Length: 121492577, Accept-Ranges: bytes
curl -sS -r -3000000 <wheel URL> -o wheeltail/tail.bin
  end of central directory: 6660 entries, central directory 943573 bytes at offset 120548982 (all in the tail)
zipnames.py tail.bin tracy capture-release csvexport cpp_device_analyses kernel_profiler lib/lib
  1731048 pjrt_plugin_tt/bin/capture-release
  1977152 pjrt_plugin_tt/bin/csvexport-release
   417816 pjrt_plugin_tt/lib/libtracy.so.0.10.0
 14214624 pjrt_plugin_tt/lib/libtt_metal.so
     6533 pjrt_plugin_tt/tt-metal/tt_metal/tools/profiler/cpp_device_analyses.json
          tracy/__init__.py, tracy/_original/*.py (the wrapped tools/tracy package)   [... trimmed ...]
zipmember.py locate tail.bin pjrt_plugin_tt/lib/libtt_metal.so      -> 101571419 5000093 14214624 8
curl -sS -r 101571419-106571712 <wheel URL>; zipmember.py inflate     -> 14214624 bytes
zipmember.py locate tail.bin pjrt_plugin_tt/lib/libtracy.so.0.10.0  -> 101385797 185555 417816 8
curl -sS -r 101385797-101571652 <wheel URL>; zipmember.py inflate     -> 417816 bytes
CRC check against the central directory: libtt_metal.so 893ecd32 = 893ecd32; libtracy.so.0.10.0 67fa94b0 = 67fa94b0
sha256: libtt_metal.so e690fe5a62efaa01bfe9085b6b1a37c64e0a5f0799846d2d509c714d2756b4db;
        libtracy.so.0.10.0 fe251a26bb5f572962cbca5ea34027095809864e9e5a7fa9eb0f2ded0fb8a7a9

grep -a -c -F <string> libtt_metal.so          (count of matching lines)
  "Invalid TT_METAL_GTEST_NUM_HW_CQS"                                   1   (rtoptions.cpp:874, any build)
  "Tracy-enabled"                                                       0   (rtoptions.cpp:675, non-Tracy builds only)
  "Device profiling never finished"                                     1   (tt_metal_profiler.cpp:853, Tracy only)
  "Debug print server is running, cannot read device profiler data"     1   (tt_metal_profiler.cpp:861, Tracy only)
  "Profiler NoC events are enabled"                                     1   (profiler.cpp:2326, Tracy only)
grep -a -c -F <string> libtracy.so.0.10.0
  "255.255.255.255" 1    "127.255.255.255" 0    "TRACY_PORT" 1    "___tracy_emit_zone_begin" 2
```

Arithmetic (one Python print, no inputs from the archive): 2 * 640**3 = 524288000; 640 * 640 * 2 = 819200 bytes
per bf16 matrix and 3 * 819200 = 2457600; 2 * 65536 * 2 + 65536 * 2 = 393216; 65536 * 2 * 2 = 262144;
51200 * 2 * 2 = 204800; 2**44 / 1e9 = 17592.19 s; 131e12 / 128 = 1.0234375e12; 524288000 / 65.5e12 = 8.00e-6 s;
524288000 / 1.0234375e12 = 5.12e-4 s.

## Sources

All read on 2026-10-26.

- https://github.com/tenstorrent/tt-metal/tree/5280a9cfb00998fd49667a29523d03aee905c129:
  docs/source/tt-metalium/tools/device_program_profiler.rst, tracy_profiler.rst; tech_reports/MetalProfiler/
  metal-profiler.md; cmake/project_options.cmake, cmake/tracy.cmake; CMakeLists.txt, tt_metal/CMakeLists.txt,
  build_metal.sh; tt_metal/llrt/rtoptions.cpp and .hpp, tt_metal/llrt/tt_cluster.cpp, tt_metal/llrt/CMakeLists.txt;
  tt_metal/jit_build/build.cpp; tt_metal/impl/profiler/{profiler.cpp, profiler.hpp, profiler_analysis.cpp,
  profiler_paths.hpp, profiler_state.hpp, profiler_state_manager.cpp, tt_metal_profiler.cpp};
  tt_metal/impl/device/firmware/profiler_initializer.cpp; tt_metal/impl/device/device_manager.cpp;
  tt_metal/distributed/mesh_device.cpp; tt_metal/api/tt-metalium/{tt_metal_profiler.hpp, experimental/profiler.hpp,
  profiler_types.hpp, host_api.hpp}; tt_metal/hostdevcommon/api/hostdevcommon/profiler_common.h;
  tt_metal/tools/profiler/{kernel_profiler.hpp, perf_counters.hpp, cpp_device_analyses.json};
  tt_metal/hw/inc/internal/tt-1xx/wormhole/tensix.h; tt_metal/impl/debug/inspector/{logger.cpp, inspector.cpp};
  tools/tracy/{__main__.py, process_ops_logs.py}; tests/tt_metal/tools/profiler/test_device_profiler.py;
  tt_metal/programming_examples/profiler/test_custom_cycle_count_slow_dispatch/.
- https://github.com/tenstorrent/tt-umd/tree/0450f1be3a21cbf17adb97517038586a6a3af4c0: device/chip/{chip.cpp,
  local_chip.cpp}, device/cluster.cpp, device/tt_device/{wormhole_tt_device.cpp, tt_sim_tt_device.cpp},
  device/simulation/{simulation_chip.cpp, tt_sim_chip.cpp}, device/firmware/{firmware_info_provider.cpp,
  wormhole_18_3_firmware_info_provider.cpp, wormhole_18_7_firmware_info_provider.cpp},
  device/api/umd/device/{types/telemetry.hpp, arch/wormhole_implementation.hpp, chip/chip.hpp,
  simulation/simulation_chip.hpp, simulation/tt_sim_chip.hpp, tt_device/tt_sim_tt_device.hpp}.
- https://github.com/tenstorrent/tt-kmd/tree/ttkmd-2.8.0: wormhole.c, telemetry.h, hwmon.c.
- https://github.com/tenstorrent/tracy/tree/0aaefbb689b4c60694edc905545fc4709fd13f6a: CMakeLists.txt,
  public/client/TracyProfiler.cpp, public/common/TracySocket.cpp.
- https://github.com/tenstorrent/tt-mlir/tree/5f396fd6ef85780bf1b6b85bef9c7d2f84a85f3c: CMakeLists.txt,
  third_party/CMakeLists.txt.
- https://github.com/tenstorrent/tt-xla/tree/ff3a67dde32ad3311258674bd3fa260e11a20e98: CMakeLists.txt,
  third_party/CMakeLists.txt, python_package/setup.py, .github/workflows/call-build.yml.
- https://github.com/tenstorrent/ttsim/blob/v1.3.4/README.md.
- https://pypi.eng.aws.tenstorrent.com/pjrt-plugin-tt/ (index; the wheel's central directory and two members by
  range request).
- https://docs.tenstorrent.com/docs-test/core/latest/_sources/aibs/wormhole/specifications.md.txt (n300s: 128 Tensix
  cores, 64 per ASIC; AI clock 1 GHz; 131 TFLOPs FP16, 466 FP8, 262 BFP8; 576 GB/s; TBP 300 W; no footnote on how
  TFLOPs are counted).
- https://docs.tenstorrent.com/tt-system-firmware/doxygen/group__telemetry__tags.html (TT-System-Firmware APIs
  19.17.99): TAG_AICLK 14, AXICLK 15, ARCCLK 16 "in megahertz"; TAG_UPDATE_TELEM_SPEED 5 in milliseconds;
  TAG_AICLK_LIMIT_MAX 63; TAG_TDP 7 in watts; TAG_ASIC_TEMPERATURE 11 in signed 16.16 fixed point; throttling tags
  56 and 75 to 77. Its applicability to the Wormhole ARC firmware of bundle 19.5.0.0 is inferred from the driver and
  UMD using the same tag numbers, not read.
- [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/pdf/2204.02311), Section 4.1 and Appendix B
  (model FLOPs utilization; 6N + 12LHQT per token, rematerialization excluded); Kaplan et al. 2020,
  https://arxiv.org/abs/2001.08361 (C about 6N per token), as PaLM attributes it.
- Repository: toolchains/tt-metal.pin, toolchains/ttsim.pin, toolchains/tt-mlir.pin; lassi/executors/tt_silicon.py
  (program_environment, :458-479), lassi/executors/sandbox.py (:20, :41-42, unshare with a private network
  namespace); assets/bench/tt-pairs-v0.yaml; assets/bench/tt-pairs-v0/*/tt/*.cpp; results/tt-silicon-smoke-20261023,
  results/p4-tt-install; tools/server/config.default.json (:32-35); plans/spikes/p9-tt-power.md and
  lassi/profilers/{tenstorrent.py, energy.py} in c:/dev/lassi (branch p9-ee, uncommitted, read only).

## Findings

### 1. The device profiler at the pin: what it records, how it is switched on, and its files

- What it is [read]: a scope-based profiler for the RISC-V cores of every Tensix core and of the Ethernet cores
  (device_program_profiler.rst:6). Firmware records default zones per RISC per program, `<RISC>-FW` (one pass of the
  firmware loop) and `<RISC>-KERNEL` (the kernel's main), and a kernel may add its own zones with
  DeviceZoneScopedN (device_program_profiler.rst:36-51, :143; metal tt_metal/tools/profiler/kernel_profiler.hpp:648).
  On a Wormhole Tensix core the RISCs are BRISC, NCRISC, and TRISC_0 to TRISC_2 (cpp_device_analyses.json:97-178;
  test_device_profiler.py:221-233).
- The timer [read]: each marker stores the 32 low bits and 12 high bits of the core's wall clock register
  RISCV_DEBUG_REG_WALL_CLOCK_L/H with a 19-bit timer id (kernel_profiler.hpp:175-179), registers at debug offsets
  0x1F0 and 0x1F8 on Wormhole (metal tt_metal/hw/inc/internal/tt-1xx/wormhole/tensix.h:148-149). So a stamp is a
  44-bit count of cycles since reset [inferred], which wraps after 2^44 cycles, about 4.9 hours at 1 GHz (PROJECTED).
  RISCs on one core share one counter; counts on different cores are "closely synced" with minor skew; counts on
  different chips are not synced (device_program_profiler.rst:163-166).
- Run-time switches [read: metal tt_metal/llrt/rtoptions.cpp, each read only when set, :1292-1300]:
  TT_METAL_DEVICE_PROFILER=1 (:669-681); TT_METAL_DEVICE_PROFILER_DISPATCH (:683-693, dispatch cores);
  TT_METAL_PROFILER_SYNC (:695-705, host-device time sync, slow dispatch only per tt_metal_profiler.cpp:601-603);
  TT_METAL_DEVICE_PROFILER_NOC_EVENTS and _RPT_PATH (:707-724); TT_METAL_PROFILE_PERF_COUNTERS (:734-756, a bitfield,
  only the FPU group supported); TT_METAL_TRACE_PROFILER, TT_METAL_PROFILER_TRACE_TRACKING (:758-780);
  TT_METAL_PROFILER_MID_RUN_DUMP (:782-792); TT_METAL_PROFILER_CPP_POST_PROCESS (:794-804);
  TT_METAL_PROFILER_SUM (:806-816); TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT (:818-826);
  TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES and _DISABLE_PUSH_TO_TRACY (:834-856). The output directory is
  TT_METAL_PROFILER_DIR, else `$TT_METAL_HOME/generated/profiler/`, else `generated/profiler/` relative to the
  working directory, with logs under `.logs/` (metal tt_metal/impl/profiler/profiler_paths.hpp:16-39).
- How kernels are instrumented [read]: with the profiler on, the JIT adds `-DPROFILE_KERNEL=<options>` and the DRAM
  buffer size to every kernel and firmware build (metal tt_metal/jit_build/build.cpp:185-201), so a profiled run
  compiles different binaries from an unprofiled one.
- When results are read [read]: MeshDeviceImpl::close_impl calls ReadMeshDeviceProfilerResults with LAST_FD_READ
  (metal tt_metal/distributed/mesh_device.cpp:729-736), and device teardown dumps every device profiler's results
  (metal tt_metal/impl/device/firmware/profiler_initializer.cpp:73-92; profiler_state_manager.cpp:88-110). A host
  program may also read earlier with ReadMeshDeviceProfilerResults (metal tt_metal/api/tt-metalium/host_api.hpp:
  601-619) or detail::ReadDeviceProfilerResults (tt_metal_profiler.hpp:52-70). All five tt-pairs-v0 TT references
  call close() on their mesh device (assets/bench/tt-pairs-v0/eltwise_binary/tt/eltwise_binary.cpp:146,
  eltwise_sfpu.cpp:142, loopback.cpp:98, matmul_multi_core.cpp:222, matmul_single_core.cpp:189) [read], so they would
  be profiled without a code change [inferred].
- Files [read]:
  - `profile_log_device.csv` (DEVICE_SIDE_LOG, profiler_paths.hpp:12). First line `ARCH: <arch>, CHIP_FREQ[MHz]:
    <aiclk>, Max Compute Cores: <n>`; header `PCIe slot, core_x, core_y, RISC processor type, timer_id,
    time[cycles since reset], data, run host ID, trace id, trace id counter, zone name, type, source line, source
    file, meta data` (profiler.cpp:1058-1066); one row per marker (profiler.cpp:1068-1128). The rst's example header
    (device_program_profiler.rst:133-134) is older and has fewer columns.
  - `cpp_device_perf_report.csv` (PROFILER_DEVICE_PERF_REPORT_NAME, profiler_paths.hpp:18), written only with
    TT_METAL_PROFILER_CPP_POST_PROCESS=1 (profiler.cpp:2168-2170, 2110-2130): one row per program execution with
    `GLOBAL CALL COUNT,METAL TRACE ID,METAL TRACE REPLAY SESSION ID,DEVICE ID,DEVICE ARCH,OP NAME,CORE COUNT,
    AVAILABLE WORKER CORE COUNT` (profiler_analysis.cpp:586-588), then one column per analysis of
    cpp_device_analyses.json (DEVICE TRACE FIRMWARE DURATION [ns], DEVICE TRACE KERNEL DURATION [ns], DEVICE KERNEL
    FIRST TO LAST START [ns], DEVICE FW DURATION [ns] with DEVICE FW START CYCLE and END CYCLE, DEVICE KERNEL DURATION
    [ns] with DEVICE KERNEL START CYCLE and END CYCLE, DEVICE KERNEL DURATION DM START [ns] with its cycles, and one
    KERNEL DURATION [ns] per RISC: cpp_device_analyses.json:1-195), then `OP TO OP LATENCY [ns],OP TO OP LATENCY
    BR/NRISC START [ns]` (profiler_analysis.cpp:601). The analysis file is read from the runtime root
    (profiler.cpp:2115-2117), which is the pinned tree, read-only, so that read works in the sandbox [inferred].
  - DEVICE KERNEL DURATION [ns] spans the first `-KERNEL` zone start to the last `-KERNEL` zone end over every core
    and RISC of one program execution (cpp_device_analyses.json:69-86; profiler_analysis.cpp:231-283).
- Program-level APIs [read]: experimental::GetLatestProgramsPerfData and GetAllProgramsPerfData return, per chip,
  each program's analyses (start and end cycle and duration in ns) with core counts, and Get*KernelDurationSummary
  gives count, min, max, mean, and a histogram of DEVICE KERNEL DURATION [ns] (metal
  tt_metal/api/tt-metalium/experimental/profiler.hpp:42-121; tt_metal_profiler.cpp:1178-1334). They return data only
  with TT_METAL_PROFILER_MID_RUN_DUMP and TT_METAL_PROFILER_CPP_POST_PROCESS set (tt_metal_profiler.cpp:937-940,
  1255-1257), and only in a Tracy build (Finding 2). LASSI does not need them: the CSVs carry the same values and are
  written by the runtime, not by the program.

### 2. The pinned install cannot use the device profiler (ENABLE_TRACY=OFF)

- [read] ENABLE_TRACY defaults ON upstream (metal cmake/project_options.cmake:7; build_metal.sh:230-233,
  :311-312), and it sets TRACY_ENABLE for the Tracy client (metal cmake/tracy.cmake:4-10). toolchains/tt-metal.pin
  builds with `-DENABLE_TRACY=OFF`, the pin's ttsim CI configuration (bible 541).
- [read] Without TRACY_ENABLE, setting TT_METAL_DEVICE_PROFILER at all raises TT_FATAL "TT_METAL_DEVICE_PROFILER
  requires a Tracy-enabled build of tt-metal." while the runtime options are read (rtoptions.cpp:673-676), so a
  program run with it aborts before it opens a device [inferred: the options are read when the MetalContext starts].
- [read] Every host-side profiler entry point compiles to an empty body without TRACY_ENABLE: InitDeviceProfiler,
  ReadDeviceProfilerResults (both forms), ReadMeshDeviceProfilerResults, ProcessDeviceProfilerResults,
  SetDeviceProfilerDir, FreshProfilerDeviceLog, GetLatest/GetAllProgramsPerfData, the interval read thread
  (tt_metal_profiler.cpp:706-771, :803-1082, :1108-1171, :1249-1344), the CSV writers and analyses
  (profiler.cpp:2072-2331, guarded at :2077, :2112, :2133, :2312), and the profiler initializer (profiler_initializer.
  cpp:30-66).
- A pitfall [read; consequence inferred]: TT_METAL_PROFILE_PERF_COUNTERS sets the profiler on without the Tracy
  check (rtoptions.cpp:751-756), and the JIT then instruments every kernel (build.cpp:185-216), while the reads are
  compiled out, so against the pinned build it would add overhead and return nothing. The tt_silicon and ttsim
  executors build the program's environment from scratch (lassi/executors/tt_silicon.py:458-479), so neither
  variable can reach a run today [read].
- Consequence: device time, cycles, and the perf counters need a tt-metal built with ENABLE_TRACY=ON, which is a
  different library (a different TRACY_ENABLE definition for the Tracy client and every guarded file), so it is a
  pin change or a second pinned install (Agent Rule 10; OQ-104 draft).

### 3. Overhead and limits

- [read] Profiling, kernel debug print, and Watcher cannot be on together (device_program_profiler.rst:29-31;
  tt_metal_profiler.cpp:860-861). The tt_silicon executor never reruns with Watcher (bible 705), so this does not
  collide there; the ttsim executor's hang rerun with Watcher is a ttsim feature only.
- [read] Each marker costs two MMIO register reads and two L1 stores on the RISC (kernel_profiler.hpp:175-179); the
  pin's own test accepts a measured loop between one fifth and ten times its reference count of 52 cycles per loop
  (test_device_profiler.py:238-257), so upstream asserts no tight overhead figure. The cycle cost per marker on these
  cards is [unknown].
- [read] Buffers: each RISC holds 250 optional markers (125 zones), 4 guaranteed markers, and 2 program ids in L1,
  2,048 bytes (profiler_common.h:85-94; device_program_profiler.rst:163); the DRAM buffer per RISC holds 48 bytes per
  program times the program support count, 1000 by default, 48,000 bytes (profiler_state_manager.cpp:19-61). When it
  fills, markers are dropped and the host logs "Profiler DRAM buffers were full, markers were dropped!"
  (profiler.cpp:1478-1490), which a reader can detect.
- [read] NoC event profiling warns of "1-15% cycle overhead" (profiler.cpp:2325-2326).
- [read] Host side: results are processed on a thread pool of up to 8 threads per device profiler (profiler_state_
  manager.cpp:113-125) at close; UMD opens the n300's second chip as a remote device in every run (results/tt-silicon-
  smoke-20261023), so two device profilers and up to 16 threads are likely [inferred], within the sandbox's tasks
  limit [inferred].
- [inferred] A profiled run's process wall time includes building instrumented firmware and kernels and dumping the
  CSVs, so it is not comparable with an unprofiled run's.

### 4. Cycles to time: the AICLK the card reports

- [read] The profiler converts cycles to nanoseconds as round(cycles x 1000 / AICLK_MHz), with AICLK from
  Cluster::get_device_aiclk at analysis time (profiler_analysis.cpp:270-276), and writes the AICLK it read at
  construction into the CSV header (profiler.cpp:2075, :1059-1062). get_device_aiclk calls the chip's get_clock
  (metal tt_metal/llrt/tt_cluster.cpp:692), which on Wormhole sends the ARC message GET_AICLK (0x34) through the open
  device (umd tt_device/wormhole_tt_device.cpp:85-96; arch/wormhole_implementation.hpp:97). It is a read inside the
  process that already holds the card, not a new device open [inferred].
- [read] The clock moves: UMD's start_device sets every chip of the cluster to BUSY and waits for AICLK to reach the
  firmware's maximum, and close sets LONG_IDLE (umd cluster.cpp:946-974; chip/chip.cpp:237-284; local_chip.cpp:
  197-202). UMD's constants are 1000 MHz busy and 500 MHz idle (umd arch/wormhole_implementation.hpp:290-291), while
  the busy target for bundle 18.7 or newer comes from the telemetry entry AICLK_LIMIT_MAX (umd firmware/
  firmware_info_provider.cpp:182-184, :43-60). The wait logs a warning, possibly overheating, when AICLK does not
  settle (chip.cpp:266-280). The vendor's n300 specification gives a 1 GHz AI clock (Sources).
- Consequences [inferred]: the ns figures assume one AICLK for the whole run; a throttled run would be converted at
  the wrong clock. So LASSI should keep raw cycles beside ns and record AICLK sampled during the run (Finding 6).
  The firmware's throttling tags (56, 75 to 77 in the firmware tag list) are not in UMD's tag enum at the pin (umd
  types/telemetry.hpp:10-60) and not in the driver's sysfs list (kmd wormhole.c:389-403) [read], so throttling is
  visible to LASSI only as a lower AICLK reading.

### 5. Tracy host profiling

- [read] Tracy is a client library in the profiled program and a server (the GUI or tracy-capture) that connects to
  it, by default on port 8086 (tech_reports/MetalProfiler/metal-profiler.md:10, :26, :34, :41). tt-metal uses a fork
  of Tracy 0.10 (metal-profiler.md:13). It profiles host code zones; with the device profiler it also shows device
  zones (device_program_profiler.rst:145-158).
- [read] The pinned build links libtracy.so.0.10.0 (toolchains/tt-metal.pin HOST_LINK_LIBRARIES; metal
  tt_metal/CMakeLists.txt:251) but with TRACY_ENABLE off, so it starts nothing.
- [read] In a Tracy-enabled build at the pin's Tracy commit, the client's options default to TRACY_ON_DEMAND off,
  TRACY_ONLY_LOCALHOST off, and TRACY_NO_BROADCAST off (tracy CMakeLists.txt:65-89); a static Profiler object is
  constructed when the library loads on Linux and spawns its worker (tracy TracyProfiler.cpp:1349, :1439); the worker
  listens on TCP port 8086, or the next free one up to 8105 when TRACY_PORT is unset (:1605-1616, :1710-1726), on
  every interface unless the environment sets TRACY_ONLY_LOCALHOST=1 (tracy TracySocket.cpp:450-462), and, without
  TRACY_NO_BROADCAST, sends a UDP announcement to 255.255.255.255:8086 every 3 s while no server is connected
  (TracyProfiler.cpp:1743-1764, :1800-1816). No environment variable turns the broadcast off at this commit; the
  runtime switches are TRACY_PORT, TRACY_NO_EXIT, TRACY_NO_SYS_TRACE, TRACY_NO_INVARIANT_CHECK, TRACY_ONLY_LOCALHOST,
  and TRACY_ONLY_IPV4 (grep of both files) [read].
- [read] LASSI's sandbox gives each run its own network namespace (lassi/executors/sandbox.py:20, :41-42), so inside
  it the listener and the broadcast cannot reach alpha01's network [inferred].
- Tracy host zones are not needed for LASSI's metrics: host time is measured by LASSI itself (the timing profiler,
  the executor's wall time, the trainer's step clock). Tracy's role here is only that the device profiler is built
  with it.

### 6. What the cards give at the pins without a build change

- Whole-run wall time [read: bible 306, 382]: Attempt.run.wall_s (the executor's run) and the timing profiler's
  runtime_s (the window around the executor's run call). On silicon it is [MEASURED] process time including device
  open, firmware and kernel JIT, transfers, the kernel, and close; the smoke's 1.8 to 1.9 s per program
  (results/tt-silicon-smoke-20261023) shows the scale. It is a measurement of the run, never of the kernel.
- Host JIT compile time [read]: tt-metal's Inspector is on by default (metal tt_metal/llrt/rtoptions.hpp:94-104;
  rtoptions.cpp:1051-1059) and writes `<logs dir>/generated/inspector/programs_log.yaml` (rtoptions.cpp:1302-1304;
  inspector/logger.cpp:62) with program_compile_started, program_kernel_compile_finished, and
  program_compile_finished entries carrying host timestamps and duration_ns (logger.cpp:114-170), plus program binary
  status changes and mesh workload events (logger.cpp:172-197, :268-370). Nothing logs a program's end on the device.
  The tt_silicon and ttsim executors already point TT_METAL_LOGS_PATH into the workdir's reserved directory
  (tt_silicon.py:476), so the file exists after a run [inferred] but is not read today. It is host time on the host
  CPU, under ttsim as on silicon.
- Card telemetry from sysfs, with no device open [read: kmd wormhole.c:389-417, :483-494, :630-676, :679-688;
  plans/spikes/p9-tt-power.md Findings 1 to 3 for the read path]:
  - `/sys/class/tenstorrent/tenstorrent!<n>/tt_aiclk`, `tt_axiclk`, `tt_arcclk`: the firmware telemetry table's
    tags 14 to 16, printed as an unsigned decimal, one ioread32 through the driver's BAR4 mapping per read; MHz per
    the firmware tag list (Sources) [unit inferred for Wormhole]. Each file is visible only when the firmware's tag
    table holds the tag (wormhole.c:483-494). Card 0 on alpha01 lists tt_aiclk (plans/spikes/tt-silicon-executor.md,
    probe rx 20261022-214312-exec-59e3) [read]; its value was not read [unknown].
  - The hwmon device under the card's PCI device: power1_input (the PCIe chip's core power, microwatts at 1 W
    resolution, OQ-093), temp1_input (legacy word 0x74, x 1000 / 16, millidegrees Celsius), in0_input (0x70,
    millivolts of VCORE), curr1_input (0x84, milliamperes) (wormhole.c:679-688) [read; units per the hwmon ABI as
    plans/spikes/p9-tt-power.md reads it].
  - tt_heartbeat, the firmware's timer heartbeat (wormhole.c:402).
  - The second chip of each n300 has no PCI function and none of these files (plans/spikes/p9-tt-power.md,
    Finding 2) [read there].
  - The refresh interval is the firmware's tag 5, which the driver does not expose (p9-tt-power.md Finding 5)
    [unknown on these cards].
- Host-timed program regions inside the tracked references [inferred, option only]: a reference could time
  EnqueueMeshWorkload through Finish with a steady clock and write it as an extra output. Under slow dispatch that
  includes host dispatch work, it changes the tracked programs, and a model's candidate could write any number there,
  so it is a weak substitute for the device profiler (OQ-104, option (d)).

### 7. tt-smi and UMD telemetry

- [read] tt-smi is not installed on alpha01 (bible 1204), the gate classes tt-smi as a tt_silicon command
  (tools/server/config.default.json:32-35), and tt-smi and luwen read telemetry by opening the card's node and
  messaging the ARC (plans/spikes/p9-tt-power.md Finding 4). So no LASSI sampler uses them (Agent Rule 9).
- [read] UMD's telemetry tags at the pin include VCORE, TDP, TDC, ASIC_TEMPERATURE, BOARD_TEMPERATURE, AICLK,
  AXICLK, ARCCLK, FAN_SPEED, AICLK_LIMIT_MAX, and TDP_LIMIT_MAX, and no INPUT_POWER (umd types/telemetry.hpp:10-60).
  They are read through an open device (p9-tt-power.md Finding 4), so only a process that already holds the card
  (a host program, a training process) could read them; LASSI's samplers read the driver's sysfs copies instead.

### 8. What ttsim reports

- [read] UMD drives ttsim through eight entry points (init, exit, PCI config and memory reads and writes, tile reads
  and writes, and clock), and advances its clock only by calling libttsim_clock(10) after each tile read
  (umd tt_device/tt_sim_tt_device.cpp:26-28, :108-121, :313; api/umd/device/tt_device/tt_sim_tt_device.hpp:71-78).
  So simulated cycles advance with the host's polling, not with wall time or with the kernel's own pace [inferred].
- [read] A ttsim chip reports AICLK 0 (umd simulation/simulation_chip.cpp:110; TTSimChip does not override it,
  api/umd/device/simulation/tt_sim_chip.hpp:18), and TTSimTTDevice::get_clock throws (tt_sim_tt_device.cpp:230-232).
  So even a Tracy-enabled tt-metal on ttsim would write CHIP_FREQ[MHz]: 0 and divide by zero in its ns conversion
  (profiler_analysis.cpp:272-275) [inferred].
- [read] ttsim's README lists "Reads from performance counters, cycle counters, or timers" among the cases that may
  diverge from silicon (ttsim README:88-91), and the bible already says timing and cycle counters are not meaningful
  there (bible 717).
- [read] The only cycle number ttsim prints is the stamp of a finding line, `[<cycle>] ERROR: <class>: ...`, for
  example `[4489] ERROR: UndefinedBehavior: decode_and_execute_csrrs: ...` (tests/executors/fixtures/ttsim/
  README.md; bible 723). That a kernel reading `mcycle` is itself UndefinedBehavior on Wormhole (the same fixture) is
  why the profiler uses the memory-mapped wall clock instead [inferred].
- Consequence: ttsim gives no device time, no cycle count usable as one, and no clock. LASSI records only the
  simulator wall time it already labels (bible 374) and the finding stamps inside the findings' text; the runner
  already refuses a profiler beside a simulator (bible 489).

### 9. FLOPs: analytic per item; Wormhole's counters measure utilization

- [read] The perf counters at the pin: groups FPU, PACK, UNPACK, L1, INSTRN, with only FPU implemented; its counters
  are FPU_COUNTER, SFPU_COUNTER, and MATH_COUNTER, each a request (busy) count over a reference cycle count (metal
  tt_metal/tools/profiler/perf_counters.hpp:9-16, :40-49, :58-86, :157-208; rtoptions.cpp:734-756), with Wormhole
  registers present (tensix.h:107, :119-121). tt-metal's report turns them into "FPU Util", "SFPU Util", and "MATH
  Util" percentages as value / ref cnt x 100 (tools/tracy/process_ops_logs.py:740-767, :114-125). They count busy
  cycles, not floating-point operations, and need the device profiler (Finding 2) [read].
- So FLOPs are analytic: a property of the item's computation, counted from the manifest's shapes, the same for the
  reference and for any candidate that solves the item (a "useful work" count, as model FLOPs are in PaLM's MFU)
  [design; arithmetic below is exact].

| Item (tt-pairs-v0) | Shapes (assets/bench/tt-pairs-v0.yaml) | Work | Minimum device bytes moved |
| --- | --- | --- | --- |
| eltwise_binary | a, b bf16 [65536]; c bf16 on the device, widened to f32 on the host | 65,536 FLOP (adds) | 393,216 (two inputs read, one output written) |
| eltwise_sfpu | src0 bf16 [65536]; result bf16 | 65,536 exp evaluations (no FLOP figure by convention) | 262,144 |
| loopback | input bf16 [51200]; output bf16 | 0 FLOP | 204,800 |
| matmul_single_core | a, b bf16 [640, 640]; c bf16 [640, 640] | 2MNK = 524,288,000 FLOP | 2,457,600 |
| matmul_multi_core | as above | 524,288,000 FLOP | 2,457,600 |

- Reference points (PROJECTED from the vendor's 131 TFLOP/s FP16 per n300 card, 128 Tensix cores, 64 per chip;
  the counting method is not stated by the vendor): 65.5 TFLOP/s per chip and about 1.02 TFLOP/s per Tensix core, so
  a 640^3 matmul needs at least about 8 microseconds on one chip's cores and about 512 microseconds on one core. The
  examples' data formats (bf16 inputs, DST_ACCUM_MODE false, bible 728) and the vendor's FP16 figure are not the
  same thing, so any percent of peak is labeled with the vendor figure it uses.

### 10. The training stack (tt-xla wheel): Tracy-enabled, with the device profiler compiled in

- [read] tt-xla at ff3a67dd sets TTMLIR_ENABLE_PERF_TRACE ON by default (xla CMakeLists.txt:46) and passes it to
  tt-mlir as TT_RUNTIME_ENABLE_PERF_TRACE (xla third_party/CMakeLists.txt:75); its wheel build passes no override
  (xla python_package/setup.py:283-306; .github/workflows/call-build.yml:150). tt-mlir passes that option to tt-metal
  as ENABLE_TRACY and installs the Tracy capture and csvexport binaries when it is on (mlir third_party/
  CMakeLists.txt:123, :306-313; mlir CMakeLists.txt:32, :93-95; read at 5f396fd6, seven commits after the wheel's
  tt-mlir c7b8bd77, the same lines assumed there).
- [read] The wheel holds pjrt_plugin_tt/bin/capture-release and csvexport-release, the wrapped tracy package, and
  cpp_device_analyses.json. Its libtt_metal.so (14,214,624 bytes, CRC matching the archive) contains strings that
  exist only inside TRACY_ENABLE blocks ("Device profiling never finished", tt_metal_profiler.cpp:853; "Profiler NoC
  events are enabled", profiler.cpp:2326) and lacks the non-Tracy abort text "Tracy-enabled" (rtoptions.cpp:675),
  while it holds another runtime-options string ("Invalid TT_METAL_GTEST_NUM_HW_CQS", rtoptions.cpp:874), which shows
  the options code is in that library (llrt is an object library of tt_metal, metal tt_metal/llrt/CMakeLists.txt:7,
  :17). Its libtracy.so.0.10.0 contains the broadcast address 255.255.255.255 and not 127.255.255.255.
- [inferred] So the wheel's tt-metal is a Tracy-enabled build: TT_METAL_DEVICE_PROFILER=1 works in a training
  process without any build, and every process that loads the wheel's libtt_metal starts Tracy's listener and UDP
  broadcast (Finding 5) unless it runs in its own network namespace. Whether the wheel loads libtracy from
  libtt_metal's dependencies at import is [inferred from the pinned link line], not read from the wheel's ELF headers.
- [inferred] Device profiling a whole training run is a poor fit: one training step runs many programs, beyond the
  1000-program DRAM buffer between reads (Finding 3), and the dumps and instrumentation change the step time. A short
  separate profiling run of a few steps is the use that fits.

### 11. Trust of device timing for model-written programs

- [inferred] The device profiler's CSVs are written by the run's own process into a directory that process can
  write, and its markers sit in L1 that the run's kernels can write. So for a model-written program the numbers are
  self-reported by the run and could be forged; for a tracked reference they are the runtime's. Today only tracked
  references run on silicon (OQ-078, OQ-099), and no reward reads silicon (Agent Rule 9), so this is a labeling rule
  for later, not a blocker now.

## Mapping onto LASSI

- Profiler (bible 224, 247; bible-p9 224, 248): a Profiler brackets the executor's run call from outside, so it can
  measure host time and sample host-side telemetry, but it cannot set the program's environment or read the run's
  directory. Device timing therefore belongs to the executor, as the simulator's findings do (RunResult.sim_ub,
  sim_gap, diagnostics, bible 235), and sysfs telemetry belongs to the Profiler, as backend `tt` already does for
  power (lassi/profilers/tenstorrent.py on p9-ee).
- Result Record (bible 273, 306, 308, 382): Attempt.run.wall_s holds process wall time; Attempt.profile holds
  runtime_s, avg_power_w, energy_j. There is no field for device time, cycles, clocks, temperature, or work.
- Reporting Rules (bible 1122-1126): "Label the device on every number; ttsim results never carry performance" and
  "Recompute every ratio from raw runtimes" already cover the principle; no rule yet says which denominator a FLOP/s
  uses or how a percent of peak is labeled.
- Executors: tt_silicon's program_environment (tt_silicon.py:458-479) sets no profiler variable and the reserved
  `@tt-silicon` directory already holds the logs (Inspector included); ttsim's environment sets none either.
- Training (bible 980, 992): checkpoint.json holds TRL's final log entry and steps.jsonl one entry per optimizer step;
  neither records a step time, a token count, or FLOPs today.

## Proposed design

### A. Device timing on silicon: an executor reading, opt-in, from a Tracy-enabled tree (needs OQ-104)

- The tt_silicon executor takes a new required setting `device_profiler`, `off` or `kernel`, with no default, as
  hugepages and reset_filter are required (bible 705). `kernel` is refused unless the executor's tree is a
  Tracy-enabled install (its lassi-install.txt and pin say so), and `off` is refused on such a tree, so no run uses
  the variant by accident. The ttsim executor never takes the setting (Agent Rule 2).
- With `kernel`, the environment gains TT_METAL_DEVICE_PROFILER=1, TT_METAL_PROFILER_CPP_POST_PROCESS=1,
  TT_METAL_PROFILER_DISABLE_PUSH_TO_TRACY=1, and TT_METAL_PROFILER_DIR=<workdir>/@tt-silicon/profiler, and nothing
  else of the profiler's (no NoC events, no perf counters, no sync, no dispatch cores).
- After the run, before the reserved directory is removed, the executor parses cpp_device_perf_report.csv (per
  program: DEVICE KERNEL DURATION [ns], DEVICE KERNEL START and END CYCLE, DEVICE FW DURATION [ns], CORE COUNT,
  AVAILABLE WORKER CORE COUNT, DEVICE ID) and the first line of profile_log_device.csv (ARCH, CHIP_FREQ[MHz]), counts
  the "markers were dropped" warnings in stderr, and keeps both CSVs in the run's binary store by sha256. A missing
  or malformed file, or dropped markers, gives a null reading with a run-stage note, never a partial number (Agent
  Rule 1).
- It returns them as RunResult.device_timing, copied to a new Attempt.run.device_timing (and Trial.reference_run
  .device_timing for the baseline's reference run): {source: "tt-metal device profiler", tree, programs:
  [{device_id, op_name, kernel_cycles, kernel_ns, fw_ns, cores}], aiclk_mhz_at_read, dropped_markers,
  self_reported}, where self_reported is true for any run of model-written code (Finding 11). Null means not
  measured: every ttsim run, every silicon run with `off`.
- Cost (PROJECTED): one more tt-metal build at the same commit, about 4 minutes and 3 GiB on alpha01 from the P4.2
  install's 3 min 44 s and 3.0 GiB (results/p4-tt-install), a ttmetal-host variant pointed at that tree, and parsing
  code tested with captured CSVs from one owner-window run of the five references.

### B. Card telemetry: sampled by the Profiler from sysfs (extends backend `tt`, no new device access)

- Backend `tt` (P9.5) already reads each card's hwmon power1_input. It would also read, at each sample, tt_aiclk and
  temp1_input of the same cards (Finding 6), and its window summary would carry them as readings: the minimum,
  maximum, and time-weighted mean of AICLK in MHz and the maximum ASIC temperature in degrees Celsius.
- The Profile record gains one optional map, `readings`: {name: {min, max, mean, unit, label}}, null when not
  measured, so the GPU backends can add their clocks the same way later; avg_power_w and energy_j keep their meaning
  and OQ-093's label.
- Agent Rule 9 is kept as p9-tt-power.md's verdict keeps it: sysfs reads, one MMIO load each by the driver, no node
  open, no ARC message. A read-only probe in an owner window confirms the values (Unknown, below).

### C. Work per item: in the suite manifest, not measured

- Each item of a TT suite declares `work`: {flop: <int or null>, ops: {<class>: <int>}, bytes: <int>}, from its
  shapes, as the table in Finding 9 gives them; flop is null for an item with no floating-point arithmetic and for
  transcendental-only items, whose counts go under ops (for example exp). Work is a [DESIGN] fact of the item, read
  from the manifest by lassi.analysis, never a measurement and never a field a run writes.

### D. Derived rates: computed in analysis, labeled, never stored as measurements

- lassi.analysis computes, per attempt run on silicon with a reading: achieved FLOP/s = work.flop / (kernel_ns x
  1e-9) and achieved bandwidth = work.bytes / (kernel_ns x 1e-9), each labeled "analytic <work> / device kernel time
  (tt-metal profiler, AICLK <x> MHz)"; and, only if OQ-105 keeps it, percent of the vendor's peak, labeled "of the
  vendor's 131 TFLOP/s FP16 per n300 (65.5 per chip), counting unstated". A rate over process wall time is computed
  only on request and labeled "over process wall time (device open, JIT, transfers, close included)".
- It refuses any rate for a run whose device names a simulator (Agent Rule 2), and any rate whose time is null.
- The reports show the raw parts beside each rate: cycles, ns, AICLK, the work count, the device line (card serial,
  firmware bundle, driver, tt-metal pin and build variant), and the date (Agent Rule 1).

### E. ttsim: what is recorded and how it is labeled

- Recorded as today: Attempt.run.wall_s and Trial.reference_run.wall_s, labeled "simulator wall time, not
  performance" (bible 374); the device line names ttsim (bible 740); finding stamps stay in the findings' text.
- Added only if wanted: Inspector's JIT compile durations from programs_log.yaml as host compile time on the host
  CPU, labeled so, since the compile runs on the host and not in the simulator. Nothing else: no device time, no
  cycles, no clock, no FLOP/s, and the profiler stays refused beside a simulator (bible 489).
- The item's work count may be shown beside ttsim results as a property of the item, never divided by any ttsim
  time.

### F. Training on the cards (task P17.38)

- The trainer records per optimizer step in steps.jsonl: step_time_s (host monotonic clock from the step's start to
  its end, after a device sync; under PyTorch/XLA that means after the step's graph is executed and waited for, which
  the first window must check [inferred]), tokens (non-pad tokens) and tokens_computed (padded tokens actually run),
  and model_flop (per step, analytic), with a flag on steps that include a graph compile (the first steps, and any
  step whose shapes are new).
- Model FLOPs per token, from the base's config.json: full weights 6N + 12LHQT (PaLM), where N counts every weight
  matrix a token passes through in a matmul (the output projection included, the embedding lookup excluded); frozen
  bases (lora, prompt_tuning) 4N + 8LHQT plus 6 x the trained parameters, since the frozen weights' gradients are not
  computed; recompute not counted. checkpoint.json records the convention's name and the summary: the median step
  time and tokens per second over the steps without a compile, and achieved model FLOP/s, labeled "analytic model
  FLOPs / measured step time".
- Device kernel time for training only in a separate profiling run of a few steps with TT_METAL_DEVICE_PROFILER=1
  (the wheel supports it, Finding 10), never across a whole run (Finding 10).
- Power and AICLK for the trained card from backend `tt` under OQ-093, if the training run is profiled.
- The device record names the card serial and the chip the run uses; the vendor peak, if reported, is per chip.

## Confidence

- High [read at the pins]: the profiler's switches, files, columns, conversion, buffers, and that it is compiled
  only with ENABLE_TRACY; that the pinned install aborts on TT_METAL_DEVICE_PROFILER; Inspector's log entries; the
  sysfs files and their units; ttsim's host-driven clock and AICLK 0; the perf counters' meaning; the Tracy
  defaults, listener, and broadcast in its source.
- High [read in the wheel]: the wheel's libtt_metal holds Tracy-only profiler strings and lacks the non-Tracy abort
  text; its libtracy holds the global broadcast address.
- Medium [inferred]: that the wheel's processes start Tracy's listener and broadcast at load; that the references
  are profiled at close without code changes; that the sysfs AICLK unit is MHz on Wormhole; that the CSVs can be
  produced inside the sandbox (read-only tree, private network).
- Unknown: everything that needs the cards (below).

## Unknown

- AICLK under load on alpha01's cards (the firmware's AICLK_LIMIT_MAX), and whether it holds during a run.
- The firmware's telemetry refresh interval (tag 5), so how fast tt_aiclk and power1_input change.
- The per-marker overhead in cycles on these cards, and the run-time cost of a profiled reference run.
- Whether a Tracy-enabled build of the pin runs cleanly in the sandbox (Tracy's threads, its listen attempt in an
  empty network namespace, its crash handler, call-stack sampling), and whether the TRACY_NO_BROADCAST and
  TRACY_ONLY_LOCALHOST configure options pass through tt-metal's add_subdirectory as the Tracy CMakeLists suggests.
- Whether PyTorch/XLA on the wheel needs an explicit wait for the device per step to time a step (Proposed design,
  section F).
- Whether the wheel's training processes actually broadcast on alpha01's network (read from the source and the
  wheel's strings; not observed).
- A read-only probe batch that settles the sysfs items, for an owner window (an rx command, so it waits for the
  owner's yes; no node is opened): for each card, `cat /sys/class/tenstorrent/tenstorrent!<n>/tt_{aiclk,axiclk,
  arcclk,heartbeat}` and the hwmon temp1_input, in0_input, curr1_input, power1_input, once at rest and once during a
  human-started reference run.

## Consequences for the plan

- Nothing changes for ttsim runs: their wall times stay labeled and no rate is computed from them.
- The tt_silicon executor's device-timing follow-up (section A) waits on OQ-104; the parsing can be built and tested
  on Windows with synthetic CSVs that follow the headers read here, then checked once against real CSVs from an
  owner-window run.
- Backend `tt` (P9.5, uncommitted on p9-ee) can add tt_aiclk and temp1_input in the same sysfs pattern, tested with
  its synthetic tree; Profile.readings is a Result Record change (Decision Log entry).
- tt-pairs-v0 gains `work` per item (section C) as a manifest change, with a test that the counts follow from the
  shapes.
- P17.38 adds the per-step fields of section F and decides network containment before the window (OQ-106).
- The Reporting Rules gain the labeling rule below.

## Proposed bible edit (the factual part)

In Execution Backends, after the executor table (after the Tenstorrent power telemetry paragraph that
plans/spikes/p9-tt-power.md proposes, if it is applied first), add:

> Tenstorrent performance telemetry, read at the pins (2026-10-26; plans/spikes/tt-perf.md; tt-metal 5280a9cf, UMD
> 0450f1be, Tracy 0aaefbb6, tt-kmd 2.8.0, ttsim v1.3.4): tt-metal's device profiler stamps zones on every RISC of
> every Tensix and Ethernet core with the core's 44-bit wall clock, records each RISC's firmware and kernel zones per
> program, and at mesh device close writes `profile_log_device.csv` (one row per marker, its first line naming the
> AICLK it read) and, with TT_METAL_PROFILER_CPP_POST_PROCESS=1, `cpp_device_perf_report.csv` (one row per program:
> DEVICE KERNEL DURATION [ns] from the first kernel start to the last kernel end over all cores, with its start and
> end cycles), under TT_METAL_PROFILER_DIR. It converts cycles to nanoseconds at the AICLK the card reports through an
> ARC message, which UMD sets to the firmware's maximum while a device is open. It is compiled only when tt-metal is
> built with ENABLE_TRACY=ON; the pinned install is built OFF, so it cannot profile, and TT_METAL_DEVICE_PROFILER=1
> aborts a program built against it. Wormhole's perf counters measure FPU, SFPU, and math busy cycles, not
> floating-point operations, and also need that build. Without it, a silicon run gives its process wall time
> (device open, JIT, transfers, kernel, and close), tt-metal Inspector's host JIT compile durations
> (`generated/inspector/programs_log.yaml` under the logs directory), and the driver's sysfs telemetry, read with no
> device open: tt_aiclk, tt_axiclk, and tt_arcclk (MHz) in the card's class directory and the hwmon temperature,
> voltage, current, and power of its PCIe chip. ttsim gives no device time: UMD advances its clock 10 cycles per host
> read and reports its AICLK as 0. FLOPs are analytic per item, never counted on the device. The tt-xla wheel
> pjrt-plugin-tt 0.9.0.dev20260223001148 carries a Tracy-enabled tt-metal at the pin, whose Tracy client listens on
> TCP 8086 and broadcasts on UDP 255.255.255.255:8086 every 3 s from library load. [OPEN] How LASSI gets device time
> (OQ-104), which FLOP convention and peak it reports (OQ-105), and how a training process holding the wheel is kept
> off alpha01's network (OQ-106); the card values (AICLK under load, the telemetry refresh interval, the profiler's
> overhead) until an owner-window probe.

In ttsim Facts, extend the bullet "Timing and cycle counters are not meaningful. No performance numbers come from
ttsim." to:

> - Timing and cycle counters are not meaningful. No performance numbers come from ttsim: UMD advances the
>   simulator's clock only after host reads (10 cycles each) and reports its AICLK as 0 (umd
>   tt_device/tt_sim_tt_device.cpp:120, simulation/simulation_chip.cpp:110 at 0450f1be; plans/spikes/tt-perf.md).

In Evaluation Protocol, Reporting Rules, add after "Label the device on every number; ttsim results never carry
performance.":

> - A Tenstorrent rate (FLOP/s, bytes/s, tokens/s) names its numerator and its denominator: work counts are analytic
>   (from the item's shapes or the model's config), never measured; times are [MEASURED] on silicon and say which
>   time (device kernel time from the tt-metal profiler with the AICLK it used, process wall time, or a training
>   step's host time). No rate is computed from a simulator run. A percent of peak names the peak figure and its
>   source.

## Proposed Decision Log entry (top of the table; add one to the total and to 2026-10-26's count)

| 2026-10-26 | What LASSI can measure on the Tenstorrent cards at the pins (Execution Backends, the Tenstorrent performance telemetry paragraph; ttsim Facts; Evaluation Protocol, Reporting Rules; Agent Rules 1, 2, and 9; plans/spikes/tt-perf.md): the pinned tt-metal, built with ENABLE_TRACY=OFF, cannot run the device profiler (TT_METAL_DEVICE_PROFILER aborts it), so silicon runs give process wall time, Inspector's host JIT compile times, and the driver's sysfs clocks, power, and temperature, read with no device open; device kernel time and cycles need a Tracy-enabled build (OQ-104). ttsim gives no device time. FLOPs are analytic per item; Wormhole's counters measure busy cycles. Rates name their numerator and denominator and are never computed from a simulator run. The tt-xla wheel's tt-metal is Tracy-enabled and its Tracy client broadcasts on the network from library load (OQ-106) | The owner asked on 2026-10-26 to track wall clock time and FLOPs on the Tenstorrent cards. Read at the pins: rtoptions.cpp:673-681 and the TRACY_ENABLE guards of tt_metal_profiler.cpp and profiler.cpp; profiler_analysis.cpp:270-276 (cycles to ns at the AICLK from get_device_aiclk); umd tt_sim_tt_device.cpp:120 and simulation_chip.cpp:110 (ttsim's clock and AICLK); perf_counters.hpp (busy counts); kmd wormhole.c:389-417 and :679-688 (sysfs); the wheel's libtt_metal.so and libtracy.so strings. No rx command ran |

## Owner-queue drafts

```
## OQ-104 Device Time On The Wormhole Cards: A Tracy-Enabled tt-metal Beside The Pinned One
State: OPEN
Kind: decision
Blocks: device kernel time and cycles for silicon runs (the tt_silicon executor's device_profiler setting, section A of plans/spikes/tt-perf.md); achieved FLOP/s worth reporting for tt-pairs-v0 on silicon
Evidence: plans/spikes/tt-perf.md, Findings 1 to 6 and section A; toolchains/tt-metal.pin (CMAKE_FLAGS -DENABLE_TRACY=OFF); results/tt-silicon-smoke-20261023 (process wall time 1.8 to 1.9 s per reference); results/p4-tt-install (3 min 44 s, 3.0 GiB)
Question: You asked to track performance on the Tenstorrent cards. At the pin, tt-metal's device profiler (kernel cycles per core and RISC, kernel duration per program, the FPU utilization counters) exists only in a build with ENABLE_TRACY=ON. LASSI's pinned install is built OFF, the configuration of the pin's ttsim CI, so it cannot profile, and TT_METAL_DEVICE_PROFILER=1 makes a program built against it abort. Without the profiler, a silicon run gives only its process wall time (about 1.8 s, nearly all device open, JIT, and close, against a kernel that would take microseconds), host JIT compile times, and sysfs clocks, power, and temperature. How should LASSI get device time?
Options:
- (a) A second install of the same commit, tt-metal@5280a9cf-tracy, built with ENABLE_TRACY=ON and Tracy's TRACY_NO_BROADCAST and TRACY_ONLY_LOCALHOST options, used only by silicon runs whose recipe sets device_profiler: kernel; ttsim and every other run keep today's install unchanged. Consequences: a pin addition (toolchains/tt-metal-tracy.pin, a Decision Log entry); one CPU-only build job on alpha01 (PROJECTED about 4 minutes and 3 GiB) and later the same on the workstation; a ttmetal-host variant for that tree; profiled runs compile instrumented firmware and kernels, so their wall times are not comparable with unprofiled runs; device time is self-reported by the run for any model-written program.
- (b) Change the one pin to ENABLE_TRACY=ON. Consequences: one tree, but every ttsim and silicon run then loads an active Tracy client (threads and a listening socket from library load), the pin leaves the ttsim CI configuration P4 qualified, and the P4 and P5 ttsim results would need a recheck on the new build.
- (c) No device profiler. Consequences: no build; silicon results carry process wall time, JIT compile time, and sysfs telemetry only, so no kernel time and no meaningful FLOP/s for these small kernels.
- (d) Time the program region inside the five tracked TT references (a steady clock around EnqueueMeshWorkload and Finish, written as an extra output). Consequences: no build; host-side time that includes slow dispatch's host work; the tracked references change; useless for model-written programs, which could write any number.
Recommendation: (a): it keeps the ttsim and correctness install exactly as qualified, confines the change to human-started silicon runs, and gives cycles and kernel time from tt-metal's own runtime. The build waits for your approval of an alpha01 job (Agent Rule 17).
Answer:
```

```
## OQ-105 What Tenstorrent Performance LASSI Reports: FLOP Counts, Peak, And Labels
State: OPEN
Kind: decision
Blocks: the work counts in tt-pairs-v0's manifest (section C of plans/spikes/tt-perf.md); derived rates in lassi.analysis and reports (section D); P17.38's per-step training metrics (section F)
Evidence: plans/spikes/tt-perf.md, Findings 9 and 10 and sections C, D, F; the vendor's n300 specification (131 TFLOP/s FP16 per card, counting unstated); PaLM, arXiv:2204.02311, Appendix B (6N + 12LHQT)
Question: Wormhole's counters measure busy cycles, not floating-point operations, so FLOPs must be counted from the work itself. Which convention should LASSI report?
Options:
- (a) Analytic useful work per item from its shapes (2MNK for a matmul, one per element for an elementwise add, exp counted apart with no FLOP figure, bytes moved for a copy), the same for every candidate of the item; rates only over a measured silicon time, naming the time (device kernel time, process wall time, or step time); a percent of peak against the vendor's 131 TFLOP/s FP16 per n300, 65.5 per chip, labeled as the vendor's figure; training as model FLOPs per PaLM (6N + 12LHQT for full weights, 4N + 8LHQT plus the trained parameters' 6N for lora and prompt_tuning, recompute not counted) over padded tokens; nothing derived from ttsim. Consequences: comparable with how GPU work is usually reported; the peak figure's counting is the vendor's and unstated.
- (b) As (a) without any percent of peak. Consequences: no reliance on an unexplained vendor figure; less context for a reader.
- (c) Times and tokens per second only, no FLOP figures. Consequences: simplest; no FLOP/s, which you asked for.
Recommendation: (a), with every rate showing its parts (work count, time, AICLK, device line) beside it.
Answer:
```

```
## OQ-106 The Training Wheel's Tracy Client On alpha01's Network
State: OPEN
Kind: policy
Blocks: training runs on the cards through the tt-xla wheel (task P17.38; OQ-101 (a)), from the window of 2026-10-27
Evidence: plans/spikes/tt-perf.md, Findings 5 and 10 (the wheel's libtt_metal.so holds Tracy-only profiler code; its libtracy.so.0.10.0 holds the broadcast address 255.255.255.255; Tracy at 0aaefbb6 starts its worker at library load, listens on TCP 8086 to 8105 on every interface, and broadcasts every 3 s, with no environment switch for the broadcast); lassi/executors/sandbox.py (private network namespace); Agent Rule 9 (alpha01's network went down twice before)
Question: Read from the wheel and the Tracy source, not observed: a training process that loads the tt-xla wheel's tt-metal would open a listening socket and send a UDP broadcast to the whole local network every 3 s for the length of the run. LASSI's sandbox would contain it (each run has its own empty network namespace); a training process outside the sandbox would not. Training needs no network after its prepare step. How should training processes run?
Options:
- (a) Each training process runs in its own network namespace with no interfaces (as the sandbox does), with the hub offline. Consequences: no listener or broadcast reaches alpha01's network; the training runner needs that wrapper and must still expose the card and hugepages as P17.38 plans.
- (b) Run as is and set TRACY_ONLY_LOCALHOST=1, which limits the listener to localhost. Consequences: the 3 s broadcast to the local network remains.
- (c) Run as is. Consequences: a listener on every interface and a broadcast every 3 s during each run; the effect on alpha01's network is not measured.
Recommendation: (a): it removes the network traffic without touching the pinned wheel, and the run needs no network.
Answer:
```
