# Spike P4.9: ttsim runtime facts, unpack_to_dest, and Watcher

- Task: P4.9 (plans/p4-ttsim.md). Bible: ttsim Facts, Execution Backends (ttsim row), Harness Contract (jit stage, timeout, Watcher), Sandbox, Risks And Questions (simulator gaps; question 4).
- Status: DONE. Phase 1 (design, local reading, read-only probes) and phase 2 (one batched job from a clean commit, the pulls, the fixtures) are complete. Values from the batch are [MEASURED] with the provenance below; values from reading the pinned source are marked (reading).
- Dates: read-only probes on 2026-09-25, 18:34 to 18:38 on the alpha01 clock (UTC-07:00). Batch run: rx job 20260926-140947-p49-ttsim-runtime-6f4c, 2026-09-26 14:09:48 to 14:15:03 (UTC-07:00), 315 s, exit status 0, 24 of 24 steps run (results/p4-ttsim-runtime/provenance.json, report/summary.txt).
- Base: branch p4-ttsim. The probes used only `rx doctor` and read-only `rx exec`, which run from the scratch root without a checkout. Batch commit: 755c739 (755c739e3d44a5a94c5f942f8f3872cc1fbb5fd3, "P4.9: Add the ttsim runtime spike batch"), a clean tree: provenance.json has dirty false and snapshot_of null, and report/batch.txt records "dirty=false, changed paths 0".
- Device: every tt-metal program ran on ttsim v1.3.4 (libttsim_wh.so, sha256 7b10aa05a5297c4a28f274e39526bfe6c69373661d1b4d4e47c4257e44b79507, a virtual Wormhole) on alpha01's CPU, with tt-metal 5280a9cfb00998fd49667a29523d03aee905c129, the joint pin (bible, Toolchain Pins; provenance.json pins). Simulator, not silicon (Agent Rule 2): every wall time here is simulator wall time, exploratory, sizing only, never performance, and ttsim's own rate line is the simulated clock rate, not performance. Every ttsim pass is provisional until confirmed on silicon. alpha01 has no Tenstorrent device node, the gate's tt_silicon class is disabled (rx doctor below), the batch's preflight refused a host with a PCI device of vendor 1e52, and no command named a device path or tool.
- Citations: "report/<file>" is results/p4-ttsim-runtime/<file> (pulled from the job's report/ directory); "local/<file>" is the job's local/ directory, pulled into the untracked .rx/pulls/local of the checkout that ran the pulls, which holds quoted upstream text and full logs and is never committed; "fixtures/<name>" is tests/executors/fixtures/ttsim/<name>.
- Output conventions: outputs are trimmed where `[... trimmed ...]` says so. Upstream source code is cited by file and line at the pin and paraphrased, not quoted (plans/p4-ttsim.md: no tt-metal file enters a tracked file). What is quoted is run-time output: the strings the examples print (such as "Test Passed"), which the checks match, and ttsim's message lines, which the P4.6 and P4.11 parsers match.

## Question

On the pinned tt-metal and ttsim, with the bible's ttsim settings: do the gate's example (add_2_integers_in_riscv) and the five upstream Tier A examples (loopback, eltwise_binary, eltwise_sfpu, matmul_single_core, matmul_multi_core) run and pass their own checks, and what do they cost (peak memory, JIT cache, simulator wall time)? What does tt-metal read, write, and need from the environment at run time, and does it run in the P0.16 sandbox with the pinned tree read-only? What do an UndefinedBehavior, a simulator gap (UnimplementedFunctionality or UnsupportedFunctionality), and a kernel JIT error look like (captured as fixtures)? Does any Tier A kernel use unpack_to_dest (ttsim issue #18)? Does Watcher work under ttsim (bible question 4)?

Why it matters: P4.6 parses the simulator's findings from these captures; P4.10 builds TT host programs and places kernels where the JIT finds them; P4.11's executor sets the settings, the sandbox allowlist, the limits, and the hang diagnostic from these facts; P4.12 may read the run's JIT cache; P4.13 takes each item's declared tolerance from its example's own check and needs each kernel's unpack_to_dest check; P4.G runs the gate's example through all of it. The Harness Contract's Watcher clause and question 4 stay [OPEN] until this answers them.

Classification: factual. Every part is answerable by reading the pinned source and running the pinned examples on ttsim. Two follow-ups the run raised are choices (Owner queue below): how the executor reads the ttsim classes the bible does not name, and whether Watcher runs on every attempt or only on a rerun of a hung one.

## Host state (rx doctor)

`uv run tools/rx.py doctor`, 2026-09-25, about 18:34 on the alpha01 clock (just before rx 20260925-183429-exec-b0ab); load and kernel lines trimmed:

```
"stop": false,
"scratch_free_gb": 582.1,
"root_free_gb": 307.7,
"host": {"hostname": "alpha01", "nproc": 256, "os": "Ubuntu 22.04.5 LTS", "mem_gb": 3169.6, "mem_avail_gb": 2880.0, [... trimmed ...]},
"config": {"min_free_gb": 5, "min_free_gb_big": 60, "max_jobs": 3, "max_big_jobs": 1, "max_build_jobs": 32},
"devices_enabled": {"rngd": true, "tt_silicon": false, "rocm_gpu": false, "nvidia_gpu": false},
"running_jobs": [],
```

scratch_free_gb is the shared disk's free space, not the scratch root's size. The job record of the batch (provenance.json) gives scratch_free_gb 582.1 before and 582.0 after, root_free_gb 302.5, nproc 256, mem_avail_gb 2985.7. The batch measured the scratch root with `du -sk`: 101204520 KiB before and 101297648 KiB after (93128 KiB more), the run directory 92340 KiB against its 6291456 KiB cap (report/checks.txt).

## Read-only findings at the pin

Three read-only probes, printing 173, 123, and 120 lines with rx's own status line (the first two exceeded the 120-line practice of plans/runs/p0-retrospective.md): rx 20260925-183429-exec-b0ab (tools, layout, examples, rtoptions names, ttsim strings), 20260925-183618-exec-1845 (kernel search, defaults, Inspector, Watcher, UMD, getenv, JIT), 20260925-183803-exec-e37d (root and logs directories, Inspector parse, simulator wait, thread pool, UMD, unpack_to_dest, compute kernels, seed targets, ttsim messages). The batch added source excerpts at the pin in local/source-excerpts.txt. Paths below are relative to the pinned tree, $LASSI_TOOLCHAINS/tt-metal@5280a9cf, unless they start with /.

### Tools on alpha01

strace 5.16 (/usr/bin/strace), GNU time (/usr/bin/time, `--version` prints "time (GNU Time) UNKNOWN"), timeout, unshare, systemd-run, prlimit, gcc, readelf, strings; no ltrace. /proc/sys/kernel/yama/ptrace_scope is 1, so strace may trace its own children. The gate's environment names no TT_* or ARCH_NAME variable; HOME is /mnt/nvme10/joseph_ufl (the scratch root), TMPDIR /mnt/nvme10/joseph_ufl/tmp, LASSI_RUNS_ROOT /mnt/nvme10/joseph_ufl/lassi-runs (exec b0ab).

### How an example finds its kernels (reading)

- The examples name kernels by relative path: tt_metal/programming_examples/CMakeLists.txt:8 defines OVERRIDE_KERNEL_PREFIX as tt_metal/programming_examples/, and each host program joins it to `<example>/kernels/...` (add_2_integers_in_riscv.cpp:94, loopback.cpp:86, eltwise_binary.cpp:123/130/135, eltwise_sfpu.cpp:82/93/101, matmul_single_core.cpp:130/141/159, matmul_multi_core.cpp:189/200/209).
- The kernel search (tt_metal/impl/kernels/kernel.cpp, the file-path lookup) tries, in order: the process's current working directory (the path is built at kernel.cpp:59); TT_METAL_KERNEL_PATH, when set (:69); the system kernel directory (:76), /usr/share/tenstorrent/kernels/ (rtoptions.cpp:253); the runtime root (:83); otherwise it throws (local/source-excerpts.txt; the phase-1 probe's :64-68 was the cwd check's branch, not the path). /usr/share/tenstorrent does not exist on alpha01 (exec 1845). [MEASURED] every seeded step compiled its copy from the step's working directory: seeded_copy_compiled is true for all eleven seeded steps, and seeded_copy_in_cache is true for every one but the two hangs, whose value is null (report/steps/seed-*.json, watcher-hang.json).
- Consequences: whatever sits at a kernel's relative path under the working directory shadows the reference kernel, so P4.10 and P4.11 can place kernel sources in the run's workdir (the sandbox's working directory), and no workdir may hold a stray tt_metal/ tree.
- The examples carry a RUNPATH into the pinned build (build_Release/tt_metal and the build tree; readelf -d, exec b0ab), so they need no loader variable.

### Runtime options (reading; tt_metal/llrt/rtoptions.cpp, rtoptions.hpp)

- Names are the entries of the EnvVarID enum (exec 1845 lists them all); the ones that matter here: TT_METAL_CACHE, TT_METAL_KERNEL_PATH, TT_METAL_LOGS_PATH, TT_METAL_SIMULATOR, TT_METAL_VISIBLE_DEVICES, TT_METAL_SKIP_DELETING_BUILT_CACHE, TT_METAL_SLOW_DISPATCH_MODE, TT_METAL_FORCE_JIT_COMPILE, TT_METAL_DISABLE_SFPLOADMACRO, TT_METAL_LOG_KERNELS_COMPILE_COMMANDS, TT_METAL_OPERATION_TIMEOUT_SECONDS, TT_METAL_WATCHER and its options, TT_METAL_INSPECTOR and its options, TT_METAL_DPRINT_*, TT_METAL_RISCV_DEBUG_INFO, TT_METAL_LLK_ASSERTS, TT_METAL_NUMA_BASED_AFFINITY. TT_METAL_RUNTIME_ROOT is read apart from the enum (:202, :263-268) and overrides the compile-time TT_METAL_INSTALL_ROOT (:257).
- TT_METAL_CACHE sets the cache directory with a tt-metal-cache component added (:369); TT_METAL_KERNEL_PATH the kernel directory (:378); TT_METAL_LOGS_PATH the logs directory (:385). The default logs directory is not settled: the batch's excerpt shows only the getter (:327) and the TT_METAL_LOGS_PATH case (:385), so the default is set elsewhere, and every run set TT_METAL_LOGS_PATH, so the default was never exercised (local/source-excerpts.txt). It does not matter for P4.11, which sets the variable.
- Inspector is on by default: rtoptions.hpp:95-108 sets enabled, rpc_server_enabled, host localhost, port 50051, and initialization_is_important false. It logs under `<logs dir>/generated/inspector` (rtoptions.cpp:1304). TT_METAL_INSPECTOR=0 turns it off (:1055-1059), TT_METAL_INSPECTOR_RPC=0 turns off its RPC server (:1142-1144), and TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS moves the server (:1115-1130). So by default every tt-metal program would open a TCP listener on localhost:50051: on a shared host that is a network surface and a collision between concurrent runs. Every batch run set TT_METAL_INSPECTOR_RPC=0, so the server's behavior in an empty network namespace was not measured. [MEASURED] with the RPC server off, Inspector itself still ran and wrote five YAML files, up to 1991 bytes each, under `<logs dir>/generated/inspector` in every run; some are empty in the aborted runs (report/files-written.txt), and the probe made no socket call (report/strace-summary.txt).
- Watcher: TT_METAL_WATCHER=<interval> enables it (:920-927); its options are TT_METAL_WATCHER_DUMP_ALL, _APPEND, _NOINLINE, _PHYS_COORDS, _TEXT_START, _SKIP_LOGGING, _DISABLE_ASSERT, _DISABLE_PAUSE, _DISABLE_RING_BUFFER, _DISABLE_STACK_USAGE, _DISABLE_SANITIZE_NOC (and the read-only and write-only L1 variants), _DISABLE_WAYPOINT, _DISABLE_DISPATCH, _ENABLE_NOC_SANITIZE_LINKED_TRANSACTION (:935-1043), plus TT_METAL_WATCHER_DEBUG_DELAY (:1340). It writes `<logs dir>/generated/watcher/` (tt_metal/impl/debug/watcher_server.cpp:94, :208, :239). No file under tt_metal/impl/debug mentions the simulator (exec 1845). The run answers whether it works (Results, Watcher).
- Simulator-specific behavior at the pin: TT_METAL_SIMULATOR makes the target device the simulator and wins over the mock cluster (rtoptions.cpp:387-405); multi-erisc mode is off with the simulator (:305); on Wormhole the context forces a teardown of active ethernet cores (tt_metal/impl/context/metal_context.cpp:356-357); the cluster type is SIMULATOR_WORMHOLE_B0 (tt_metal/llrt/tt_cluster.cpp:51-97); the SoC descriptor path comes from the simulator path (tt_metal/llrt/core_descriptor.cpp:55); and the host's wait for cores to finish has no timeout under the simulator (tt_metal/llrt/llrt.cpp:316-320: an infinite wait). [MEASURED] the seeded hang ran to its 120 s limit and was stopped there (Results, Seeded captures).
- Variables read by getenv outside rtoptions (literal names, exec 1845): TT_LOGGER_FILE, TT_LOGGER_LEVEL, TT_LOGGER_TYPES, TT_METAL_LOGGER_FILE, TT_METAL_LOGGER_LEVEL, TT_METAL_LOGGER_TYPES, TT_METAL_HOME, TT_METAL_THREADCOUNT, TT_METAL_CCACHE_KERNEL_SUPPORT, TT_METAL_CQ_SIZE_OVERRIDE, TT_METAL_KERNEL_READBACK_ENABLE, TT_METAL_PROFILER_DIR, TT_METAL_JIT_ANALYTICS, TT_METAL_RECORD_NOC_TRANSFER_DATA, TT_METAL_RISCV_DEBUG_INFO, TT_METAL_WATCHER_DEBUG_DELAY, TT_VISIBLE_DEVICES, TT_MESH_ID, TT_MESH_HOST_RANK, TT_SIMULATOR_LOCALHOST, NNG_SOCKET_ADDR, NNG_SOCKET_LOCAL_PORT, TT_BACKEND_CPUSET_ALLOCATOR_* (three), CI, CONTINUOUS_INTEGRATION, GITHUB_ACTIONS. Which of these a run actually asks for: Results, What tt-metal reads, writes, and needs.
- The JIT (tt_metal/jit_build/build.cpp): a failed compile or link throws after reading its log file (:53-60), the log being `<object>.log` or `<elf>.log` in the cache (:483-486, :564-566), so the jit-stage diagnostic is in the exception text; the cache root's other entries are removed at start (:72; PHASE-NOTES P4); the default cache root is under HOME, or /tmp without HOME (:90, :92; PHASE-NOTES P4 cites :86-93); TT_METAL_CCACHE_KERNEL_SUPPORT makes the JIT use ccache (:130).
- Thread pool: tt_metal/common/executor.hpp:12-14 sizes the executor from TT_METAL_THREADCOUNT when set, else from std::thread::hardware_concurrency(), and :20 builds one static pool of that size (local/source-excerpts.txt). On alpha01's 256 CPUs that is 256 threads, which with the main thread passes the sandbox's TasksMax of 256 (Results, The P0.16 sandbox).
- UMD (tt_metal/third_party/umd/device): its robust mutexes are named shared-memory objects created with shm_open, prefix TT_UMD_LOCK. (utils/robust_mutex.cpp:51, :258-260), which LocalChip creates for PCIe chips (chip/local_chip.cpp:119, :134); warm reset names a listener directory on the root filesystem, /tmp/tt_umd_listeners (api/umd/device/warm_reset.hpp:63, warm_reset.cpp); a simulator path ending in .so loads the library in process and takes soc_descriptor.yaml from its directory (simulation/simulation_chip.cpp:23, :32-33), while TT_SIMULATOR_LOCALHOST and the NNG variables belong to the socket-based simulator host (simulation/simulation_host.cpp:56). [MEASURED] no step left anything in its private /tmp, /var/tmp, or /dev/shm: each listing holds nothing or only /run/mount, a directory the mount tooling makes in the private /run (report/files-written.txt), and /tmp/tt_umd_listeners stayed absent on the host (report/checks.txt).

### ttsim v1.3.4 (libttsim_wh.so strings)

- 813 strings of 6 or more characters (exec e37d). The words UndefinedBehavior, UnimplementedFunctionality, and UnsupportedFunctionality each appear once, beside the format `[%ld] ERROR: %s: %s%s`. [MEASURED] a finding prints as one stdout line `[<n>] ERROR: <class>: <function>: <message>`, such as `[4489] ERROR: UndefinedBehavior: decode_and_execute_csrrs: Wormhole does not support Zicsr`; n is a simulated cycle count (ttsim's exit line `[4600] 0.4 seconds (12.1 KHz)` in the gate's clean run matches 4600 cycles at that rate). The probe searched the strings for those three words only; the run also printed NonContractualBehavior, a class the probe did not look for. A symbol decode_and_execute_unimplemented is present; [MEASURED] its finding is classed UndefinedBehavior, not UnimplementedFunctionality.
- Message strings: "Wormhole does not support Zicsr"; "fence.i does not flush the instruction cache on babyrisc and should not be used"; "fence instructions do not enforce memory ordering on Wormhole and should not be used"; "could not decode instruction inst=0x%x at pc=0x%llx"; "unaligned addr=0x%llx size=%d"; "unaligned target_pc=0x%llx"; "invalid pc=0x%llx"; "invalid x=%d"; "invalid y=%d"; "TLB region overrun: offset=0x%x size=%d"; "bar0 overrun ..."; "bar0: misaligned offset=0x%x"; "DRAM mmap() failed"; "SrcA bank is not valid" and its SrcB and already-valid variants; "pipe %d inst fifo full"; and a speed report, "%.1f seconds (%.1f MHz)" or "(%.1f KHz)" [... trimmed ...]. Which message carries which class: Results, Seeded captures.
- No string names getenv or a TTSIM variable, so the library reads no environment variable of its own (inferred from the strings; not measured). It exports libttsim_init, libttsim_exit, libttsim_clock, libttsim_pci_config_rd32/wr32, libttsim_pci_mem_rd_bytes/wr_bytes, libttsim_pci_dma_mem_rd_bytes/wr_bytes, libttsim_set_pci_dma_mem_callbacks, libttsim_tile_rd_bytes/wr_bytes, and ttsim_rv32_get_core_active/set_core_active.
- The ttsim README (https://github.com/tenstorrent/ttsim, read 2026-09-26, paraphrased) lists more classes than the bible names: UndefinedBehavior, UnpredictableValueUsed, and NonContractualBehavior, which it defines by the tt-isa-documentation glossary; UntestedFunctionality (implemented but not enabled for lack of tests); UnimplementedFunctionality (not yet implemented, planned); UnsupportedFunctionality (unlikely to be implemented without strong justification); SystemError and ConfigurationError (OS, command line, environment, or configuration problems); and AssertionFailure (an internal simulator bug). The glossary (https://github.com/tenstorrent/tt-isa-documentation/blob/main/Glossary.md, read 2026-09-26, paraphrased) defines non-contractual behavior as behavior that works on current hardware but is not part of the architectural contract, and undefined behavior as behavior the specification guarantees nothing about.

### unpack_to_dest in the Tier A references (reading)

| Example | Kernels | Compute calls (compute kernel) | CB formats in the host program | ComputeConfig |
| --- | --- | --- | --- | --- |
| add_2_integers_in_riscv (gate) | one data-movement kernel | none | none named | none |
| loopback | one data-movement kernel | none | none named | none |
| eltwise_binary | reader, writer, compute tiles_add.cpp | binary_op_init_common, add_tiles_init, add_tiles, pack_tile | Float16_b (3 mentions) | math_fidelity HiFi4 only (eltwise_binary.cpp:137) |
| eltwise_sfpu | reader, writer, compute eltwise_sfpu.cpp | init_sfpu, exp_tile_init, copy_tile, exp_tile, pack_tile | Float16_b (2) | math_fidelity HiFi4 (eltwise_sfpu.cpp:103-104); no other field the probe's pattern matched |
| matmul_single_core | reader, writer, compute mm.cpp | mm_init, matmul_tiles, pack_tile | Float16_b (1) | math_fidelity and compile_args (matmul_single_core.cpp:161) |
| matmul_multi_core | reader, writer, compute mm.cpp | mm_init, matmul_tiles, pack_tile | Float16_b (1) | math_fidelity and empty compile_args (matmul_multi_core.cpp:211) |

Sources: exec b0ab (kernel paths, DataFormat mentions, and every line naming fp32_dest_acc_en, unpack_to_dest, UnpackToDest, ComputeConfig, or MathFidelity), exec e37d (the calls in each compute kernel). The JIT passes the host's unpack_to_dest_mode to the generated descriptors (tt_metal/jit_build/genfiles.cpp:268-273, :395; tt_metal/api/tt-metalium/kernel_types.hpp:80).

Reading: no Tier A host program sets fp32_dest_acc_en or unpack_to_dest_mode, and every circular buffer is Float16_b, so no reference takes the Float32 unpack path of ttsim issue #18; two references have no compute kernel at all. The ComputeConfig defaults at the pin: fp32_dest_acc_en false (kernel_types.hpp:78) and unpack_to_dest_mode an empty vector (:80), whose element type offers UnpackToDestFp32 and Default (base_types.hpp:42); program_descriptors.hpp:89-91 has the same defaults (local/source-excerpts.txt). The run's evidence is under Results, unpack_to_dest.

### The examples' own checks (reading)

These are the declared tolerances P4.13 takes (plans/p4-ttsim.md, Tier A); local/source-excerpts.txt holds the lines.

- loopback: a pass flag (loopback.cpp:30) cleared on a mismatch (:134, :141) after a size check (:127); prints "Test Passed" (:150), else throws "Test Failed" (:152). Exact equality.
- eltwise_binary: a size check (eltwise_binary.cpp:164) and a per-element comparison with an absolute tolerance of 1e-2 (eps, :163; the comparison :169), which prints "Result mismatch at index ..." and clears the flag (:170-171); "Test Passed" (:186), else "Test Failed" (:188).
- eltwise_sfpu: a per-element comparison with an absolute tolerance of 2e-2 (:145, :151-153); "Test Passed" (:168), else "Test Failed" (:170).
- matmul_single_core and matmul_multi_core: the PCC of the bfloat16 result against a CPU golden (check_bfloat16_vector_pcc), printed as "Metalium vs Golden -- PCC = <value>" and required to exceed 0.97 (matmul_single_core.cpp:239-241, matmul_multi_core.cpp:332-334); "Test Passed" (:253, :346).
- add_2_integers_in_riscv: it does check its result, which the phase-1 probe's pattern (pass, fail, mismatch, PCC, TT_FATAL, TT_THROW, EXIT_FAILURE, throw) missed: it prints "Error: Expected result vector size of 1, got ..." (:124) or "Error: Expected result of 21, got ..." (:129), else "Success: Result is <value>" (:134). Whether it exits nonzero after an Error line is not in the excerpt. So P4.11's smoke driver can require the line "Success: Result is 21" beside exit status 0. The batch's own check pattern did not know that line, so report/runs.tsv says "none printed" for every run of this example; the logs hold the line (local/logs).

## The batch run

Files, committed at 755c739: plans/spikes/p4-ttsim-runtime/run.sh (the orchestrator), spike.py (stdlib helpers: measure, seed, quiet, strace and getenv summaries, summarize), sandbox_attempt.py (runs one example through lassi.executors.sandbox), getenv_log.c (an LD_PRELOAD shim that logs which variables a process asks for, never their values). Each file's header states its contract.

Command, from the clean commit in the detached clean worktree (plans/LESSONS.md, alpha01), with `rx doctor` before and after:

```
uv run tools/rx.py doctor
uv run tools/rx.py job start --big --name p49-ttsim-runtime --timeout 10800 -- 'bash plans/spikes/p4-ttsim-runtime/run.sh'
uv run tools/rx.py job wait <id> --timeout 600 --interval 300 -n 120
uv run tools/rx.py doctor
```

The explicit `--timeout` matters: rx job start otherwise sends 86400 s (tools/rx.py:444), which the gate clamps to 172800, so a runaway would get a day. The batch's stdout stayed under 120 lines; the step table is in report/summary.txt.

Stopping it (not needed in this run): do not use `rx job kill`. It kills the runner and frees the slot (tools/server/gate.py:752-764) while bash is still reading run.sh from that slot, so the batch may die at its next write, skipping the checks and the summary. Touch the per-run stop file instead, which the batch checks between steps and (through spike.py measure --stop-file) during a step: `uv run tools/rx.py exec -- 'touch /mnt/nvme10/joseph_ufl/lassi-runs/p49-ttsim-runtime/<rx id>/STOP'`. The owner's gate STOP is honored too, and a shared $LASSI_RUNS_ROOT/p49-ttsim-runtime/STOP only while it is newer than the batch's start.

Output under $LASSI_RUNS_ROOT/p49-ttsim-runtime/20260926-140947-p49-ttsim-runtime-6f4c/:

- report/: text evidence, ASCII only; pulled into results/p4-ttsim-runtime/.
- fixtures/: every seeded and Watcher capture, ASCII only; pulled into the untracked .rx/pulls/fixtures, and six copied into tests/executors/fixtures/ttsim/ (Results, Seeded captures).
- local/: logs capped at 1 MiB, the JIT-generated descriptor lines, and source excerpts, which quote upstream text; pulled only into the untracked .rx/pulls/local.
- raw/: JIT caches, working directories, seeded kernel copies, the strace log; never pulled. The whole run directory is 92340 KiB (report/checks.txt).

Pulls, from the checkout that commits results/, with .rx/pulls/report, fixtures, and local absent before each `--path` pull (rx extracts into .rx/pulls/<basename> and `--into` copies everything there):

```
uv run tools/rx.py pull <id> --into results/p4-ttsim-runtime
uv run tools/rx.py pull --path lassi-runs/p49-ttsim-runtime/<id>/report --into results/p4-ttsim-runtime
uv run tools/rx.py pull --path lassi-runs/p49-ttsim-runtime/<id>/fixtures
uv run tools/rx.py pull --path lassi-runs/p49-ttsim-runtime/<id>/local
```

results/p4-ttsim-runtime/summary.md cites provenance.json and the report files.

Steps, in order (report/runs.tsv). The acceptance items ran first and the extras last, so a tight budget would drop an extra; the budget was not reached. Each step had its own wall limit: spike.py measure stops the step's process group at the limit (SIGTERM, then SIGKILL 10 s later), and an inner `timeout -k 10` 30 s later stops it even if the helper has died.

1. Clean runs of the six examples with the bible's settings (TT_METAL_SIMULATOR, the descriptor beside it, TT_METAL_SLOW_DISPATCH_MODE=1, TT_METAL_DISABLE_SFPLOADMACRO=1), plus TT_METAL_RUNTIME_ROOT, TT_METAL_CACHE and TT_METAL_LOGS_PATH inside the step, a HOME of the step's own, LANG=C with LC_ALL=C, and TT_METAL_INSPECTOR_RPC=0; 900 s each; /usr/bin/time -v for peak memory; a once-a-second sampler for peak tasks and summed resident set. The gate's example passed with loopback down, so the loopback fallback did not run.
2. The probe: the gate's example under strace (file, network, and exec calls) with the getenv shim and no HOME. It passed at once, so the HOME rerun did not run.
3. Seeded failures on copies of the gate example's kernel, one inserted line each (spike.py SEEDS), found through the step's working directory and also named by TT_METAL_KERNEL_PATH, compile commands logged: three UB candidates (an unaligned 4-byte load, `fence`, `fence.i`), four gap candidates (a CSR read, an undecodable word, `ecall`, `wfi`), a call to an undeclared function (the JIT error), and a wait on an L1 word nothing writes (the hang). Every seed ran line-buffered (stdbuf -oL -eL).
4. Watcher: the gate's example with TT_METAL_WATCHER=1, then the seeded hang with it.
5. The sandbox: the gate's example through lassi.executors.sandbox (SandboxSpec with $LASSI_SCRATCH and the runs root hidden, $LASSI_TOOLCHAINS read-only, the workdir under the run, Limits cpus 16, memory 8192 MiB, disk 2048 MiB, wall 60 s, TasksMax 256), a HOME retry after the fast failure, then a TT_METAL_THREADCOUNT=16 retry, then eltwise_binary with the thread count that passed. At this commit ENVIRONMENT_NAMES holds no TT_METAL_* name, so each example got its variables through its argv (`env -i NAME=value... <example>`, with LANG=C and LC_ALL=C), and the attempt recorded SandboxSpec's refusal of the same variables as SandboxSpec.environment.
6. Extras (not acceptance): eltwise_sfpu without TT_METAL_DISABLE_SFPLOADMACRO, and one unbuffered control seed.

The measuring runs (steps 1, 2, 3, 4, 6) ran outside the sandbox, as plans/p4-ttsim.md allows for P4.9's measuring runs. The seeded lines are hand-written reference edits, never model code, run only on ttsim's simulated cores under an unmodified upstream host program (Agent Rule 6). Each measuring run was confined by:

- `unshare -rmnipf --kill-child --mount-proc`: its own user, mount, network, IPC, and pid namespaces, loopback down, the host network never offered; the whole process tree dies with the namespace.
- Private tmpfs mounts on /tmp, /var/tmp, /dev/shm, and /run, listed before they went away.
- Read-only recursive binds (--rbind, made read-only with mount_setattr) of $LASSI_TOOLCHAINS (both pinned trees), the TurboQuant checkout, and the shared default JIT cache root; a `.ready` marker records that the wrapper reached the program.
- `env -i` for the whole environment, with LANG=C and LC_ALL=C; the gate's locale is en_US.UTF-8, under which GCC diagnostics carry UTF-8 quotes (bible, Host Facts). spike.py writes every file under report/ and fixtures/ as ASCII.
- prlimit --core=1 (one byte), --fsize (2 GiB per file), and --as (1024 GiB per process) on the whole chain; a 2 GiB cap on each step's stdout plus stderr.
- Its own JIT cache.

The core limit was set at the top with `prlimit --pid $$ --core=1:1` (bash's `ulimit -c 1` is 1024 bytes, which does not disable a piped systemd-coredump), and preflight verified getrlimit(RLIMIT_CORE) = (1, 1) (report/batch.txt: "core limit (RLIMIT_CORE): 1 byte, set with prlimit --core=1:1 and verified by getrlimit"). So the jit-error seed's abort and the two sandbox aborts stored no core on the root filesystem (Agent Rule 7; the OQ-014 incident). In the probe, strace's first child is a shell that always exits normally, since strace re-raises a child's fatal signal after setting its own RLIMIT_CORE to 0. After the run, rx 20260926-142038-exec-5ccd (`ls -lt /var/lib/systemd/coredump`, read-only) found no core from the batch: its three newest files are root-owned cores of VLLM::Worker processes from 2026-09-25. The listing is recorded in OQ-034's Evidence (plans/OWNER-QUEUE.md), not in results/.

Local stub dry run before the launch (exploratory; 2026-09-25 and 2026-09-26; a local Linux environment with stand-ins for the examples, strace, and uv; no remote activity): all 24 steps ran and every path was exercised. It tested the script's logic, not tt-metal or ttsim.

PROJECTED before the run, not measured: 30 to 90 minutes and under 2 GiB of scratch. Measured: 315 s of job time (provenance.json) and a 92340 KiB run directory (report/checks.txt).

Batch shortcomings found in the evidence, none of which changes a finding:

- spike.py's class pattern knew only UndefinedBehavior, UnimplementedFunctionality, and UnsupportedFunctionality, so seed-ub-unaligned and its unbuffered control have an empty sim_classes; the step's sim_lines holds the NonContractualBehavior line (report/steps/seed-ub-unaligned.json). A parser should take any `ERROR: <Word>:` class, not a fixed list.
- unpack_facts counted a descriptor line as enabled when it held a nonzero digit, and the lines it read carried a leading occurrence count, so every run with a descriptor reads any_unpack_to_dest_enabled true (report/unpack_to_dest-facts.txt; report/summary.txt says "enabled in" nine runs). The lines themselves (local/unpack_to_dest.txt) say otherwise: Results, unpack_to_dest.
- The check pattern did not know the gate example's "Success: Result is 21" line (above).

What the run could not answer: behavior on silicon (Agent Rule 2); performance; the reads, writes, and environment needs of a compute-kernel example beyond what the eltwise_binary sandbox run shows (the probe traced only add_2_integers_in_riscv); classes no seed reached (UnimplementedFunctionality among them); the sandbox with the TT variables in ENVIRONMENT_NAMES (they went through argv; the route differs, not the values); Tier A items P4.13 writes itself; concurrency between runs; Inspector's RPC server in an empty network namespace; and behavior with the pinned trees writable.

## Results

Every value in this section is [MEASURED 2026-09-26: rx job 20260926-140947-p49-ttsim-runtime-6f4c from commit 755c739 (clean), ttsim v1.3.4 with tt-metal 5280a9cf on alpha01's CPU, a simulator; results/p4-ttsim-runtime/] unless marked otherwise. One run per step, so no value has a spread.

### Clean runs (simulator wall time, exploratory, sizing only)

From report/runs.tsv and report/steps/clean-*.json; the own-check column from the logs (local/logs/clean-*.stdout.txt). Peak RSS is GNU time's maximum resident set of one process; summed RSS is the process group's total, sampled once a second (it counts shared pages once per process, so it overstates the group's use); tasks are the process group's peak, and every clean run's program peaked at 257 threads (steps/clean-*.json peak_process_threads).

| Example | Exit | Own check | ttsim messages | Peak RSS (MiB) | Summed RSS (MiB) | Peak tasks | JIT cache (KiB / files) | Wall (s, simulator) | ttsim exit line |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| add_2_integers_in_riscv | 0 | passed: "Success: Result is 21" | none | 98 | 60 | 267 | 2996 / 70 | 1.39 | [4600] 0.4 seconds (12.1 KHz) |
| loopback | 0 | passed: "Test Passed" | none | 100 | 80 | 269 | 3100 / 70 | 1.42 | [7660] 0.5 seconds (15.8 KHz) |
| eltwise_binary | 0 | passed: "Test Passed" | none | 114 | 162 | 273 | 6088 / 120 | 1.64 | [9870] 0.5 seconds (21.1 KHz) |
| eltwise_sfpu | 0 | passed: "Test Passed" | none | 108 | 204 | 296 | 6260 / 120 | 1.49 | [10360] 0.5 seconds (19.5 KHz) |
| matmul_single_core | 0 | passed: PCC 0.9806516 (> 0.97) | none | 104 | 382 | 276 | 5980 / 120 | 3.73 | [366200] 2.7 seconds (135.7 KHz) |
| matmul_multi_core | 0 | passed: PCC 0.99999344 (> 0.97) | none | 104 | 434 | 285 | 6036 / 120 | 2.9 | [14550] 1.9 seconds (7.8 KHz) |

"ttsim messages: none" means no `ERROR:` line in the run's output (local/logs). Each run also logged UMD's warning that the simulated n150 board's harvesting mask disagrees with the board type, and two Fabric warnings that it could not create `<runtime root>/generated/fabric` on the read-only tree (below); neither changed a result.

### What tt-metal reads, writes, and needs at run time

From the probe (probe-add2: the gate's example under strace, no HOME, exit 0, "Success: Result is 21"; report/strace-summary.txt, 79011 traced lines) and the getenv shim (report/getenv.txt), plus every step's listing (report/files-written.txt).

- Reads: the pinned tree (2612 successful reads and 5217 successful lookups; most under runtime/sfpi/compiler, tt_metal/hw/inc, runtime/hw/lib, tt_metal/third_party/tt_llk, tt_metal/hostdevcommon/api), the ttsim directory (11 reads: the library and soc_descriptor.yaml), /lib (277, shared libraries), /etc (only /etc/ld.so.cache and /etc/localtime), /sys (21), /proc, and /dev/urandom, the only /dev entry. 32119 failed lookups in the pinned tree, most likely the compiler trying each include directory (inferred, not itemized); 4810 failed lookups under /mnt outside the scratch root and 4874 elsewhere under the scratch root are not attributed by the summary (all failed lookups, none a read or write).
- Writes: the step's cache and logs (344 successful writes in the run directory) and the private /tmp (342, the compiler's temporary files, removed before the step ended). The one write outside them: a failed mkdir of `<runtime root>/generated` (report/strace-summary.txt: "tt-metal@5280a9cf/generated (mkdir failed)"). The logs show it is the Fabric control plane exporting chip-mapping files to `<runtime root>/generated/fabric` at device open; on the read-only tree it logs two warnings and continues (local/logs/clean-add_2_integers_in_riscv.stdout.txt). Every run that compiled its kernels left Inspector's five YAML files and Watcher's kernel_names.txt and kernel_elf_paths.txt in its logs directory, even without Watcher (the two aborted sandbox runs left only Inspector's files); seeded runs also held their kernel copy in the working directory; no HOME directory received a file (report/files-written.txt).
- Executables started: the example, /usr/bin/env (the batch's env -i), the pinned sfpi compiler (riscv-tt-elf-g++ 33 times, GCC 15.1.0's cc1plus, lto1, lto-wrapper, collect2, as, and ld under runtime/sfpi/compiler), /usr/bin/make (16), and /bin/sh (18). So the JIT needs make and sh from the host system.
- Network calls: none (report/strace-summary.txt, empty network section).
- Environment: the example asked for 156 distinct names (report/getenv.txt), and only the seven the batch set were set: TT_METAL_CACHE, TT_METAL_DISABLE_SFPLOADMACRO, TT_METAL_INSPECTOR_RPC, TT_METAL_LOGS_PATH, TT_METAL_RUNTIME_ROOT, TT_METAL_SIMULATOR, TT_METAL_SLOW_DISPATCH_MODE. It asked for TT_METAL_THREADCOUNT (4 times), TT_METAL_HOME, TT_METAL_KERNEL_PATH, TT_METAL_VISIBLE_DEVICES, and the logger, Watcher, DPRINT, profiler, and debug-delay names, all unset. No process the shim logged (the example and the GCC processes; make, sh, and as do not appear in its log) asked for HOME: with TT_METAL_CACHE set, the traced run needed no HOME. The compiler's processes asked for LANG, PATH, TMPDIR (set) and GCC's own names.
- Host checks (report/checks.txt): "root-fs check: passed; this account changed nothing there since the start (59 unreadable, 0 raced)"; "pinned trees: 0 path(s) changed since the start in tt-metal@5280a9cf and ttsim@v1.3.4"; the default cache root under the shared HOME, 0 top entries changed; /tmp/tt-metal-cache and /tmp/tt_umd_listeners absent.
- Single chip: tt-metal's topology search logged `target_graph_nodes=1, global_graph_nodes=1` and opened one user-mode driver; UMD described one chip, an n150 board (local/logs). No variable is needed to keep it to one chip.

### The P0.16 sandbox, pinned tree read-only

From report/steps/sandbox-*.json, report/summary.txt, and the helpers' logs (local/logs/sandbox-*). Limits: cpus 16, memory 8192 MiB, disk 2048 MiB, wall 60 s, TasksMax 256; $LASSI_TOOLCHAINS read-only; the scratch root and the runs root hidden.

| Attempt | TT_METAL_THREADCOUNT | Exit | Hang | Killed | program_s (simulator wall time, sizing only) | Workdir complete | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sandbox-add2 | unset | 134 | no | no | 0.09 | yes | aborted at start: `terminate called after throwing an instance of 'std::system_error'`, `what():  Resource temporarily unavailable`; empty JIT cache |
| sandbox-add2-home | unset | 134 | no | no | 0.09 | yes | the same, with HOME=<workdir> |
| sandbox-add2-threads16 | 16 | 0 | no | no | 1.32 | yes | "Success: Result is 21"; JIT cache 2996 KiB / 70 files |
| sandbox-eltwise_binary | 16 | 0 | no | no | 1.42 | yes | "Test Passed"; JIT cache 6088 KiB / 120 files |

Why the gate's example aborted: tt-metal's executor pool is sized by TT_METAL_THREADCOUNT, else by std::thread::hardware_concurrency() (executor.hpp:12-14), which is 256 on alpha01 (provenance.json nproc 256); every measuring run's program but three peaked at 257 threads (the pool and the main thread; steps/*.json peak_process_threads), one more than the sandbox's TasksMax of 256; the other three were the two Watcher runs at 258 and the JIT-error run, which aborted early, at 144. So a thread creation fails with EAGAIN, which libstdc++ throws as std::system_error "Resource temporarily unavailable", and the uncaught exception aborts the program. A HOME changes nothing; TT_METAL_THREADCOUNT=16 passes. The mechanism is the reading plus these three runs; no run traced the failing pthread_create itself.

SandboxSpec.environment with the TT variables: refused, as expected (report/summary.txt): "the sandbox environment may not set 'TT_METAL_RUNTIME_ROOT'; allowed names: LANG, LC_ALL, NVHPC_CUDA_HOME, OMP_NUM_THREADS, PATH, TMPDIR". P4.11 adds the ttsim row's names to that list (plans/p4-ttsim.md).

### Seeded captures (fixtures)

From report/runs.tsv, report/summary.txt, report/steps/seed-*.json, and local/logs. Every ttsim finding is the last line of the run's stdout, stderr stays empty, and the program exits 1; the host program prints nothing after it. Every seeded copy was compiled from the step's working directory (seeded_copy_compiled true in each step's record).

| Seed (inserted line) | Class ttsim gave | ttsim line | Exit | Fixture |
| --- | --- | --- | --- | --- |
| ub-unaligned (4-byte load at an address 2 mod 4) | NonContractualBehavior | `[4500] ERROR: NonContractualBehavior: rv32_mem_rd: unaligned addr=0x16dfe2 size=4` | 1 | non-contractual-behavior/ |
| ub-fence (`fence iorw, iorw`) | UnsupportedFunctionality | `[4489] ERROR: UnsupportedFunctionality: decode_and_execute_fence: fence instructions do not enforce memory ordering on Wormhole and should not be used` | 1 | unsupported-functionality/ |
| ub-fencei (`fence.i`) | UnsupportedFunctionality | `[4489] ERROR: UnsupportedFunctionality: decode_and_execute_fence_i: fence.i does not flush the instruction cache on babyrisc and should not be used` | 1 | none |
| gap-zicsr (`csrrs a0, mcycle, zero`) | UndefinedBehavior | `[4489] ERROR: UndefinedBehavior: decode_and_execute_csrrs: Wormhole does not support Zicsr` | 1 | undefined-behavior/ |
| gap-decode (the word 0xffffffff) | UndefinedBehavior | `[4489] ERROR: UndefinedBehavior: decode_and_execute_unimplemented: could not decode instruction inst=0xffffffff at pc=0x8238` | 1 | none |
| gap-ecall (`ecall`) | UnsupportedFunctionality | `[4489] ERROR: UnsupportedFunctionality: decode_and_execute_ecall_ebreak: inst=0x73` | 1 | none |
| gap-wfi (`wfi`) | UnsupportedFunctionality | `[4489] ERROR: UnsupportedFunctionality: decode_and_execute_ecall_ebreak: inst=0x10500073` | 1 | none |
| jit-error (a call to an undeclared function) | none: a kernel JIT error | stdout: tt-metal's critical log line `TT_THROW: brisc build failed. Log: ...`; stderr: `terminate called after throwing an instance of 'std::runtime_error'`, then the compiler's `reader_writer_add_in_riscv.cpp:6:5: error: 'lassi_seeded_jit_error' was not declared in this scope` | 134 (SIGABRT) | jit-error/ |
| hang (a wait on an L1 word nothing writes) | none | only ttsim's progress lines, `[10000000] 23.2 seconds (430.2 KHz)` to `[50000000] 114.4 seconds (437.0 KHz)` | 137: stopped at its 120 s limit (timed_out true, 131.0 s) | hang/ |

So the seeds meant as UB and gap candidates landed partly the other way round from their names: the fences, ecall, and wfi are UnsupportedFunctionality; the CSR read and the undecodable word are UndefinedBehavior; the unaligned load is NonContractualBehavior, a class the bible does not name; and no seed produced UnimplementedFunctionality. The exit status of a stopped step (137) is the measuring chain's after the SIGKILL, not the program's own. A kernel JIT error aborts the host program, since the JIT's exception is uncaught in the example; the diagnostic is in the exception text on stderr and in tt-metal's log line on stdout, plain ASCII under LC_ALL=C.

Buffering: the unbuffered control (seed-ub-unaligned-unbuffered, no stdbuf) printed the same NonContractualBehavior line on stdout and exited 1 (local/logs), so a ttsim finding is not lost without line buffering.

Fixtures: the six captures above, copied from the pulled fixtures into tests/executors/fixtures/ttsim/<name>/ (stdout.txt, stderr.txt, status.json; watcher-hang/ also watcher.log), with captures.json (the sha256 of each pulled and each tracked file) and a README giving their provenance. report/fixture-check.txt found 0 compiler source-context lines and 0 compile-command lines in every capture. The copy applied one rule by code on top of the batch's sanitizing: each backtrace frame line (starting with " --- ") became the batch's marker line, 31 in jit-error/stderr.txt and 15 in jit-error/stdout.txt, none elsewhere. The other captures (ub-fencei, gap-decode, gap-ecall, gap-wfi, the unbuffered control, watcher-add2) stay in the untracked pulls.

### unpack_to_dest (from the run)

The JIT-generated descriptor lines in each clean run's cache (local/unpack_to_dest.txt, three distinct lines per run): every run's descriptors set DST_ACCUM_MODE to false (the line occurs once in each data-movement example's cache and three times in each compute example's), and no line names unpack_to_dest at all (report/unpack_to_dest-facts.txt: unpack_to_dest_lines 0 in every run); the other two lines open the unpack_src_format and unpack_dst_format arrays, whose values were not kept. No run printed a ttsim `ERROR:` line, so the issue's `tensix_unpacr` UndefinedBehavior did not occur. report/unpack_to_dest-facts.txt's any_unpack_to_dest_enabled true is the batch's false positive (Batch shortcomings above): the "enabled" line it counted is `DST_ACCUM_MODE = false` with an occurrence count in front.

Finding: no Tier A reference uses unpack_to_dest or Float32 dest accumulation at the pin, from the host programs (no fp32_dest_acc_en or unpack_to_dest_mode set, both off by default, every CB Float16_b), the generated descriptors, and the runs (all six pass with no ttsim error). Issue #18's path was not exercised, so whether it occurs at the pin for a kernel that takes it stays untested; P4.13 checks each new item.

### TT_METAL_DISABLE_SFPLOADMACRO

eltwise_sfpu without it (sfplm-on-eltwise_sfpu): exit 0, "Test Passed", no ttsim error, JIT cache 6260 KiB / 120 files, the same as the clean run with it (report/runs.tsv). At the pin it changes no measured result for the one example that uses the SFPU. The row keeps it as the bible has it (the CI job at the pin did not set it; plans/spikes/p4-tt-pins.md).

### Watcher

From report/summary.txt, report/steps/watcher-*.json, report/files-written.txt, local/logs, and fixtures/watcher-hang/watcher.log.

| Step | Exit | watcher.log | Hung core's state in the log | Wall (s, simulator) |
| --- | --- | --- | --- | --- |
| watcher-add2 | 0, "Success: Result is 21" | 26526 bytes, 222 lines, 2 dumps | n/a | 2.98 (clean run: 1.39) |
| watcher-hang | 137: stopped at its 120 s limit (timed_out true) | 1637305 bytes, 12945 lines, 128 dumps | worker core (0,0), BRISC: `R` (running the kernel) in dump 1, `NSW` (a NOC semaphore wait) in every later dump kept (80 of 80 later dumps, 81 of 128 kept: dumps 1 to 41 and 89 to 128 in the 1 MiB local copy); k_ids 1, which the log maps to the seeded kernel's path | 131.06 |

Watcher works under ttsim at the pin: with TT_METAL_WATCHER=1 the program still passes, Watcher writes `<logs dir>/generated/watcher/watcher.log`, and each dump lists every core's waypoints per RISC-V (BRISC, NCRISC, TRISC0-2), the kernel ids, and the id-to-path map, about once a second of simulator wall time (about 12.8 KB per dump on this simulated chip). The hang's kept dumps name the hung core, the RISC-V, the wait, and the kernel, which is what the Harness Contract's hang diagnostic wants. Costs, simulator wall time and exploratory: the gate's example took 2.98 s with Watcher against 1.39 s without (one run each). Watcher also builds its debug code into the kernels (tt-metal's Watcher documentation; not read at the pin), so a Watcher run is not quite the program a plain run is.

### Host checks

Root filesystem: passed, this account changed nothing under /tmp, /var/tmp, or /dev/shm (59 entries unreadable, 0 raced); pinned trees: 0 paths changed; default cache roots: 0 top entries changed in the shared HOME's tt-metal-cache, /tmp/tt-metal-cache and /tmp/tt_umd_listeners absent; scratch du before 101204520 KiB, after 101297648 KiB; run directory 92340 KiB against its 6291456 KiB cap (report/checks.txt). No core from the batch (rx 20260926-142038-exec-5ccd; the listing is in OQ-034's Evidence, above).

## Settings the bible's ttsim row lacks

Each confirmed by the run, with the ttsim row's existing settings kept.

1. TT_METAL_RUNTIME_ROOT on the pinned tree and TT_METAL_CACHE inside the run: already required by Toolchain Pins (Install) and PHASE-NOTES P4, absent from the row. The runtime root can stay read-only: the only write tt-metal tries there is the Fabric export, which fails with a warning and does not stop the run.
2. TT_METAL_LOGS_PATH inside the run: Inspector (with its RPC server off) and the kernel lists write under `<logs dir>/generated/` in every run, and Watcher's log goes there too. Without it they go to a default the run did not exercise.
3. TT_METAL_INSPECTOR_RPC=0: every run set it; the program then made no socket call. Inspector's own YAML files are small (under 2 KiB each) and harmless. TT_METAL_INSPECTOR=0 was not tried, so the row names the measured setting.
4. TT_METAL_THREADCOUNT: required in the sandbox. Unset, the pool of one thread per host CPU (256 on alpha01) passes TasksMax 256 and the program aborts; with 16 both sandboxed examples pass. The row sets it to the run's Limits.cpus, as the native executor sets OMP_NUM_THREADS (bible, Sandbox).
5. HOME: not needed. No process the getenv shim logged asked for it with TT_METAL_CACHE set, the probe passed without it, and the sandboxed runs passed without it too.
6. Single chip: ttsim presents one chip (one topology node, one n150 board); no variable is needed.
7. TT_METAL_DISABLE_SFPLOADMACRO: no measured difference at the pin; kept.
8. LANG=C and LC_ALL=C: the kernel compiler is a GCC, whose diagnostics carry UTF-8 quotes under a UTF-8 locale (bible, Host Facts); under LC_ALL=C the jit-error capture is plain ASCII. The native runs' LANG=C.UTF-8 (bible, Sandbox) is a UTF-8 locale too, so it would not do for ttsim.

## Finding

On ttsim v1.3.4 with tt-metal 5280a9cf, with the bible's settings plus the runtime root, a JIT cache and logs directory in the run, Inspector's RPC server off, and LC_ALL=C, the gate's example and all five Tier A examples exit 0 and pass their own checks, no Tier A reference uses unpack_to_dest, and Watcher works (a hung kernel's dumps show its core at a NOC semaphore wait), so question 4 is answered yes. ttsim reports a finding as one `[<cycle>] ERROR: <class>: <function>: <message>` stdout line and exits 1 (UndefinedBehavior for a CSR read and an undecodable word, UnsupportedFunctionality for fence, fence.i, ecall, and wfi, and NonContractualBehavior, a class the bible does not name, for an unaligned load), a kernel JIT error aborts the host program with status 134, a hang runs to the wall limit, and in the P0.16 sandbox the program aborts unless TT_METAL_THREADCOUNT keeps tt-metal's thread pool under TasksMax 256.

Confidence: high for the exit statuses, own checks, classes, JIT abort, hang, Watcher, unpack_to_dest, and the sandbox result, each read from logs of one clean run per step; high for the thread-pool cause (source reading, a peak of 257 threads in every measuring run but three, the two Watcher runs at 258 and the JIT-error run at 144, and the 16-thread retry passing), though the failing thread creation was not traced; medium for the reads and writes, which come from one traced example without a compute kernel (the sandboxed eltwise_binary run covers a compute example's needs only as far as it passed); none for timing, which is simulator wall time from one run each and sizes limits only.

## Consequences for the plan

Firm from the reading and the run:

- P4.10 and P4.11: the kernel search looks in the working directory first (kernel.cpp:59), so kernel sources placed at their relative path in the workdir are found, and the executor must keep any other tt_metal/ tree out of a workdir.
- P4.11: the executor always sets the row's settings, with the additions above: TT_METAL_RUNTIME_ROOT (read-only), TT_METAL_CACHE and TT_METAL_LOGS_PATH in the workdir, TT_METAL_INSPECTOR_RPC=0, TT_METAL_THREADCOUNT=<Limits.cpus>, LANG=C, and LC_ALL=C, and no HOME. Every name joins the sandbox's fixed allowlist (the Sandbox edit the plan names). The sandbox must offer /usr/bin/make and /bin/sh, which the JIT runs (it does today: the sandboxed runs passed).
- P4.11 limits: one process peaked at 114 MiB and a process group's sampled sum at 434 MiB; the sandboxed runs passed under MemoryMax 8192 MiB, so 8192 MiB is ample for the Tier A references. A run's JIT cache is 3 to 6.1 MiB in 70 to 120 files, and its logs a few KiB (a Watcher log grows about 12.8 KB per dump, about once a second: 1.6 MB for a 130 s hang, so a Watcher log needs a cap before it is kept). All sizes are from the references; model kernels may differ.
- P4.11 and P4.6: parse findings from stdout with `^\[\d+\] ERROR: (\w+): ` taking any class name; a finding ends the program with exit status 1. The parser test fixtures are tests/executors/fixtures/ttsim/. A kernel JIT error arrives as an abort (134) with `brisc build failed` (or the failing RISC-V's name) in the exception text: the executor reclassifies it as stage `jit` from that text (Harness Contract).
- P4.6 and P4.11: the bible names UndefinedBehavior (fed back) and UnimplementedFunctionality and UnsupportedFunctionality (sim-gap). The run shows NonContractualBehavior as a fourth class, and ttsim's README lists more. How the executor reads them was a choice, filed as OQ-036: option (a) is applied under the owner's direction of 2026-09-27 and P4.11 builds it (NonContractualBehavior and UnpredictableValueUsed read as undefined behavior; UntestedFunctionality, SystemError, ConfigurationError, AssertionFailure, and any unknown class read as a gap whose message names the class).
- P4.6: `fence`, `fence.i`, `ecall`, and `wfi` in a kernel give UnsupportedFunctionality, so under the bible's rule a model kernel that uses one ends its trial at sim-gap with no correction and, under df-v0, no computed R. OQ-036 keeps that for now (option (a)) and records option (b), reading such a finding from the model's own kernel as undefined behavior, as a follow-up for P7's reward work.
- P4.11: under the simulator a hang runs until the executor's wall limit (llrt.cpp:316-320; measured), so the hang diagnostic comes from the executor. Watcher works; whether it runs on every attempt or on a rerun of a hung attempt was a choice, filed as OQ-037: option (b), a hung attempt rerun once with Watcher under the same wall limit, is applied under the owner's direction of 2026-09-27 and P4.11 builds it.
- P4.11: `rx job kill` stops only the gate's runner, so any long batch that runs ttsim programs needs its own stop mechanism, as this one had.
- P4.11 smoke driver (tools/ttsim_smoke.py): add_2_integers_in_riscv passes when it exits 0 and prints "Success: Result is 21".
- P4.12: the JIT cache holds the kernel ELFs and the generated descriptors under `<TT_METAL_CACHE>/tt-metal-cache/`; the descriptor lines (DST_ACCUM_MODE and the unpack format arrays) are where a kernel's dest-accumulation mode can be read.
- P4.13: declared tolerances from the examples' own checks: loopback exact; eltwise_binary absolute 1e-2; eltwise_sfpu absolute 2e-2; matmul_single_core and matmul_multi_core PCC above 0.97 against a CPU golden; add_2_integers_in_riscv exact (21). Each Tier A item's unpack_to_dest note: none takes the path at the pin (above).
- Rule 7: a tt-metal abort must run with a 1-byte core limit (prlimit --core=1:1), as here; the sandbox's own core handling is P0.16's.

## Proposed bible edit

The bible edit is one script run by the task that holds docs/BIBLE.md, after P4.10's and P4.15's edits land; it re-reads the Decision Log count at apply time. It covers:

- Execution Backends, ttsim row, Key settings: add TT_METAL_RUNTIME_ROOT (read-only), TT_METAL_CACHE and TT_METAL_LOGS_PATH in the workdir, TT_METAL_INSPECTOR_RPC=0, TT_METAL_THREADCOUNT=<Limits.cpus>, LANG=C, and LC_ALL=C. Status stays Planned (the registered ttsim Executor is P4.11; in that table Status means the executor is available).
- ttsim Facts: the issue #18 bullet's "not verified" sentence replaced by the Tier A result; a new block of [MEASURED] facts with this run's provenance: the six examples' results, the finding format and the class of each seeded condition (NonContractualBehavior included, its reading [OPEN] with OQ-036 until P4.11 builds the applied option), the JIT abort, the infinite wait, Watcher, what a run needs, reads, and writes, the thread pool in the sandbox, and SFPLOADMACRO.
- Harness Contract, timeout bullet: "if Watcher works under ttsim ([OPEN])" replaced by the finding, with the run mode [OPEN] with OQ-037 until P4.11 builds the applied option.
- Risks And Questions: question 4 answered; the simulator-gaps row's mitigation notes the Tier A unpack_to_dest check.
- Decision Log: two rows dated 2026-09-26, and the count sentence raised by two.

## Owner queue

Two choices the run raised are filed in plans/OWNER-QUEUE.md as OQ-036 and OQ-037, which the bible edit cites. Under the owner's direction of 2026-09-27 (take the recommendation of every open item not answered personally, apply it, and flag it for review in the next phase), OQ-036 option (a) and OQ-037 option (b) are applied, and P4.11 builds them; each item closes with that commit. The items as filed, for reference:

A (OQ-036, How The ttsim Executor Reads Finding Classes). How the ttsim executor reads the classes the bible does not name, and gap classes raised by model code. Kind: decision. Blocks: P4.11 (parser mapping). Evidence: this spike (Results, Seeded captures), tests/executors/fixtures/ttsim/, the ttsim README. Question: the bible reads UndefinedBehavior as a failed run fed back to the model and UnimplementedFunctionality and UnsupportedFunctionality as sim-gap. ttsim also printed NonContractualBehavior (an unaligned load), and its README lists UnpredictableValueUsed, UntestedFunctionality, SystemError, ConfigurationError, and AssertionFailure. Separately, `fence`, `fence.i`, `ecall`, and `wfi` in a kernel give UnsupportedFunctionality, so a model kernel using one ends the trial at sim-gap with no correction and no computed df-v0 R. Options: (a) NonContractualBehavior and UnpredictableValueUsed read as undefined behavior (sim_ub, fed back); UntestedFunctionality, SystemError, ConfigurationError, and AssertionFailure as gaps; an unknown class as a gap whose message names it; UnsupportedFunctionality stays a gap. Consequence: the model gets feedback on contract violations, simulator faults never count against it, and a model's fence still ends the trial. (b) As (a), but UnsupportedFunctionality from an instruction in the model's own kernel (fence, fence.i, ecall, wfi) reads as undefined behavior, fed back as a failed run. Consequence: closes the path by which a model's kernel ends a trial without a score; needs the executor to tell which kernel ran the instruction (the finding names no kernel; the only kernel in the run, or Watcher's k_id, would have to supply it), which is new design work in P4.11. (c) Every class other than UndefinedBehavior reads as a gap. Consequence: simplest; contract violations never reach the model. Recommendation: (a) now, and (b) recorded as a follow-up for P7's reward work, when a trained policy could exploit it; (a) is what the definitions say and needs no kernel attribution.

B (OQ-037, When The ttsim Executor Runs Watcher). Whether Watcher runs on every ttsim attempt or on a rerun of a hung one. Kind: decision. Blocks: P4.11 (the Watcher dump). Evidence: this spike (Results, Watcher). Question: the Harness Contract adds a Watcher dump to the hang diagnostic, and Watcher works under ttsim; it must be enabled when the program starts, so the executor either (a) runs every ttsim attempt with TT_METAL_WATCHER=1, or (b) reruns an attempt that hung once with it, under the same wall limit, and appends the last dump. Consequences: (a) costs every run Watcher's overhead (the gate's example 2.98 s against 1.39 s of simulator wall time, one run each) and its debug code in the kernels (not read at the pin), so the scored run is not quite the program the bible's settings describe, but gives the dump without a second run; (b) leaves every scored run uninstrumented and costs a second full wall limit (at least 30 s) per hang, on a shared host. Recommendation: (b), since only hangs pay, scored runs stay as specified, and ttsim runs are deterministic enough that the seeded hang hung the same way with Watcher; its log is capped (about 12.8 KB per dump).

No deletion is recommended: the whole run directory on the host is 92340 KiB.

## Sources

- alpha01: rx doctor (2026-09-25, about 18:34, UTC-07:00); rx 20260925-183429-exec-b0ab, 20260925-183618-exec-1845, 20260925-183803-exec-e37d (read-only probes, 2026-09-25); rx job 20260926-140947-p49-ttsim-runtime-6f4c from commit 755c739 (the batch, 2026-09-26), pulled into results/p4-ttsim-runtime/ (provenance.json, batch.txt, runs.tsv, summary.txt, steps/*.json, strace-summary.txt, getenv.txt, files-written.txt, checks.txt, unpack_to_dest-facts.txt, fixture-check.txt) and the untracked .rx/pulls/fixtures and .rx/pulls/local; rx 20260926-142038-exec-5ccd (the core check after the run).
- The pinned tree, $LASSI_TOOLCHAINS/tt-metal@5280a9cf (tt-metal 5280a9cfb00998fd49667a29523d03aee905c129; results/p4-tt-install/summary.md): tt_metal/llrt/rtoptions.cpp, rtoptions.hpp, llrt.cpp, tt_cluster.cpp, core_descriptor.cpp; tt_metal/impl/kernels/kernel.cpp; tt_metal/impl/context/metal_context.cpp; tt_metal/impl/debug/watcher_server.cpp, dprint_server.cpp, noc_logging.cpp, inspector/data.cpp; tt_metal/jit_build/build.cpp, genfiles.cpp; tt_metal/common/executor.hpp; tt_metal/api/tt-metalium/kernel_types.hpp, base_types.hpp, program_descriptors.hpp; tt_metal/programming_examples/CMakeLists.txt and the six examples' host programs and kernels; tt_metal/third_party/umd/device/utils/robust_mutex.cpp, chip/local_chip.cpp, api/umd/device/warm_reset.hpp, warm_reset.cpp, simulation/simulation_chip.cpp, simulation/simulation_host.cpp.
- $LASSI_TOOLCHAINS/ttsim@v1.3.4/libttsim_wh.so (sha256 7b10aa05a5297c4a28f274e39526bfe6c69373661d1b4d4e47c4257e44b79507): its strings.
- plans/spikes/p4-tt-pins.md (issue #18, the CI facts at the pin), plans/spikes/p0-sandbox-hardening.md and lassi/executors/sandbox.py (the sandbox), plans/PHASE-NOTES.md (P4: the JIT cache rule).
- ttsim issue #18: https://github.com/tenstorrent/ttsim/issues/18 (as read for P4.1 on 2026-09-24; not reread here).
- ttsim README, error classes: https://github.com/tenstorrent/ttsim (read 2026-09-26). tt-isa-documentation glossary: https://github.com/tenstorrent/tt-isa-documentation/blob/main/Glossary.md (read 2026-09-26).
