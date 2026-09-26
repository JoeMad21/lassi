# Spike P4.9: ttsim runtime facts, unpack_to_dest, and Watcher

- Task: P4.9 (plans/p4-ttsim.md). Bible: ttsim Facts, Execution Backends (ttsim row), Harness Contract (jit stage, timeout, Watcher), Sandbox, Risks And Questions (simulator gaps; question 4).
- Status: DRAFT. Phase 1 (design, local reading, read-only probes) is done; every value the batch run fills is marked PLACEHOLDER. Nothing below is [MEASURED] until the batch runs from a clean commit.
- Dates: read-only probes on 2026-09-25, 18:34 to 18:38 on the alpha01 clock (UTC-07:00). Batch run: PLACEHOLDER (rx job id, start, end).
- Base: branch p4-ttsim. The probes used only `rx doctor` and read-only `rx exec`, which run from the scratch root without a checkout, so no repository commit took part. Batch commit: PLACEHOLDER (clean tree, run from a detached clean worktree).
- Device: the batch runs every tt-metal program on ttsim v1.3.4 (libttsim_wh.so, a virtual Wormhole) on the host CPU, with tt-metal 5280a9cf, the joint pin (bible, Toolchain Pins). Simulator, not silicon (Agent Rule 2): wall times here are simulator wall times, exploratory, sizing only, never performance, and ttsim's own rate line is the simulated clock rate, not performance. alpha01 has no Tenstorrent device node, the gate's tt_silicon class is disabled (rx doctor below), the batch's preflight refuses a host with a PCI device of vendor 1e52, and no command named a device path or tool.
- Output conventions: outputs are trimmed where `[... trimmed ...]` says so. Upstream source code is cited by file and line at the pin and paraphrased, not quoted (plans/p4-ttsim.md: no tt-metal file enters a tracked file). What is quoted is run-time output: the strings the examples print (such as "Test Passed"), which the checks match, and ttsim's message strings, which the P4.6 and P4.11 parsers match.

## Question

On the pinned tt-metal and ttsim, with the bible's ttsim settings: do the gate's example (add_2_integers_in_riscv) and the five upstream Tier A examples (loopback, eltwise_binary, eltwise_sfpu, matmul_single_core, matmul_multi_core) run and pass their own checks, and what do they cost (peak memory, JIT cache, simulator wall time)? What does tt-metal read, write, and need from the environment at run time, and does it run in the P0.16 sandbox with the pinned tree read-only? What do an UndefinedBehavior, a simulator gap (UnimplementedFunctionality or UnsupportedFunctionality), and a kernel JIT error look like (captured as fixtures)? Does any Tier A kernel use unpack_to_dest (ttsim issue #18)? Does Watcher work under ttsim (bible question 4)?

Why it matters: P4.6 parses the simulator's findings from these captures; P4.10 builds TT host programs and places kernels where the JIT finds them; P4.11's executor sets the settings, the sandbox allowlist, the limits, and the hang diagnostic from these facts; P4.12 may read the run's JIT cache; P4.13 takes each item's declared tolerance from its example's own check and needs each kernel's unpack_to_dest check; P4.G runs the gate's example through all of it. The Harness Contract's Watcher clause and question 4 stay [OPEN] until this answers them.

Classification: factual. Every part is answerable by reading the pinned source and running the pinned examples on ttsim. One follow-up may become a choice once the run measures Watcher's cost (see Owner queue).

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

scratch_free_gb is the shared disk's free space, not the scratch root's size. The scratch root held 101197976 KiB after P4.2 (results/p4-tt-install/summary.md); the batch records `du -sk` before and after: PLACEHOLDER.

## Read-only findings at the pin

Three read-only probes, printing 173, 123, and 120 lines with rx's own status line (the first two exceeded the 120-line practice of plans/runs/p0-retrospective.md): rx 20260925-183429-exec-b0ab (tools, layout, examples, rtoptions names, ttsim strings), 20260925-183618-exec-1845 (kernel search, defaults, Inspector, Watcher, UMD, getenv, JIT), 20260925-183803-exec-e37d (root and logs directories, Inspector parse, simulator wait, thread pool, UMD, unpack_to_dest, compute kernels, seed targets, ttsim messages). Paths below are relative to the pinned tree, $LASSI_TOOLCHAINS/tt-metal@5280a9cf, unless they start with /.

### Tools on alpha01

strace 5.16 (/usr/bin/strace), GNU time (/usr/bin/time, `--version` prints "time (GNU Time) UNKNOWN"), timeout, unshare, systemd-run, prlimit, gcc, readelf, strings; no ltrace. /proc/sys/kernel/yama/ptrace_scope is 1, so strace may trace its own children. The gate's environment names no TT_* or ARCH_NAME variable; HOME is /mnt/nvme10/joseph_ufl (the scratch root), TMPDIR /mnt/nvme10/joseph_ufl/tmp, LASSI_RUNS_ROOT /mnt/nvme10/joseph_ufl/lassi-runs (exec b0ab).

### How an example finds its kernels (reading)

- The examples name kernels by relative path: tt_metal/programming_examples/CMakeLists.txt:8 defines OVERRIDE_KERNEL_PREFIX as tt_metal/programming_examples/, and each host program joins it to `<example>/kernels/...` (add_2_integers_in_riscv.cpp:94, loopback.cpp:86, eltwise_binary.cpp:123/130/135, eltwise_sfpu.cpp:82/93/101, matmul_single_core.cpp:130/141/159, matmul_multi_core.cpp:189/200/209).
- The kernel search (tt_metal/impl/kernels/kernel.cpp, the file-path lookup) tries, in order: the process's current working directory (kernel.cpp:64-68); TT_METAL_KERNEL_PATH, when set (:69-75); the system kernel directory (:76), /usr/share/tenstorrent/kernels/ (rtoptions.cpp:253); the runtime root (:83); otherwise it throws. /usr/share/tenstorrent does not exist on alpha01 (exec 1845). The batch confirms the line numbers in local/source-excerpts.txt and, per seed, records seeded_copy_in_cache as direct proof.
- Consequences: whatever sits at a kernel's relative path under the working directory shadows the reference kernel, so P4.10 and P4.11 can place kernel sources in the run's workdir (the sandbox's working directory), and no workdir may hold a stray tt_metal/ tree. The batch's seeded captures use exactly this: a copy of the kernel under the step's working directory, the pinned tree untouched.
- The examples carry a RUNPATH into the pinned build (build_Release/tt_metal and the build tree; readelf -d, exec b0ab), so they need no loader variable.

### Runtime options (reading; tt_metal/llrt/rtoptions.cpp, rtoptions.hpp)

- Names are the entries of the EnvVarID enum (exec 1845 lists them all); the ones that matter here: TT_METAL_CACHE, TT_METAL_KERNEL_PATH, TT_METAL_LOGS_PATH, TT_METAL_SIMULATOR, TT_METAL_VISIBLE_DEVICES, TT_METAL_SKIP_DELETING_BUILT_CACHE, TT_METAL_SLOW_DISPATCH_MODE, TT_METAL_FORCE_JIT_COMPILE, TT_METAL_DISABLE_SFPLOADMACRO, TT_METAL_LOG_KERNELS_COMPILE_COMMANDS, TT_METAL_OPERATION_TIMEOUT_SECONDS, TT_METAL_WATCHER and its options, TT_METAL_INSPECTOR and its options, TT_METAL_DPRINT_*, TT_METAL_RISCV_DEBUG_INFO, TT_METAL_LLK_ASSERTS, TT_METAL_NUMA_BASED_AFFINITY. TT_METAL_RUNTIME_ROOT is read apart from the enum (:202, :263-268) and overrides the compile-time TT_METAL_INSTALL_ROOT (:257).
- TT_METAL_CACHE sets the cache directory with a tt-metal-cache component added (:369); TT_METAL_KERNEL_PATH the kernel directory (:378); TT_METAL_LOGS_PATH the logs directory (:385). The default logs directory when TT_METAL_LOGS_PATH is unset: PLACEHOLDER (the batch records every logs_dir_ line of rtoptions.cpp in local/source-excerpts.txt).
- Inspector is on by default: rtoptions.hpp:95-108 sets enabled, rpc_server_enabled, host localhost, port 50051, and initialization_is_important false. It logs under `<logs dir>/generated/inspector` (rtoptions.cpp:1304). TT_METAL_INSPECTOR=0 turns it off (:1055-1059), TT_METAL_INSPECTOR_RPC=0 turns off its RPC server (:1142-1144), and TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS moves the server (:1115-1130). So by default every tt-metal program would open a TCP listener on localhost:50051: on a shared host that is a network surface and a collision between concurrent runs (TurboQuant's tt-metal may use the same port), and inside the sandbox's empty network namespace it has no interface to bind (the effect there: PLACEHOLDER). The batch sets TT_METAL_INSPECTOR_RPC=0 in every run and runs every measuring step in a network namespace of its own.
- Watcher: TT_METAL_WATCHER=<interval> enables it (:920-927); its options are TT_METAL_WATCHER_DUMP_ALL, _APPEND, _NOINLINE, _PHYS_COORDS, _TEXT_START, _SKIP_LOGGING, _DISABLE_ASSERT, _DISABLE_PAUSE, _DISABLE_RING_BUFFER, _DISABLE_STACK_USAGE, _DISABLE_SANITIZE_NOC (and the read-only and write-only L1 variants), _DISABLE_WAYPOINT, _DISABLE_DISPATCH, _ENABLE_NOC_SANITIZE_LINKED_TRANSACTION (:935-1043), plus TT_METAL_WATCHER_DEBUG_DELAY (:1340). It writes `<logs dir>/generated/watcher/` (tt_metal/impl/debug/watcher_server.cpp:94, :208, :239). No file under tt_metal/impl/debug mentions the simulator (exec 1845), so nothing at the pin turns Watcher off or changes it for ttsim; whether it works is the batch's question.
- Simulator-specific behavior at the pin: TT_METAL_SIMULATOR makes the target device the simulator and wins over the mock cluster (rtoptions.cpp:387-405); multi-erisc mode is off with the simulator (:305); on Wormhole the context forces a teardown of active ethernet cores (tt_metal/impl/context/metal_context.cpp:356-357); the cluster type is SIMULATOR_WORMHOLE_B0 (tt_metal/llrt/tt_cluster.cpp:51-97); the SoC descriptor path comes from the simulator path (tt_metal/llrt/core_descriptor.cpp:55); and the host's wait for cores to finish has no timeout under the simulator (tt_metal/llrt/llrt.cpp:311-320: timeout_ms is set to 0, an infinite wait). So a kernel that deadlocks on ttsim never returns by itself: the executor's wall limit is the only stop, as the Harness Contract's timeout rule already provides.
- Variables read by getenv outside rtoptions (literal names, exec 1845): TT_LOGGER_FILE, TT_LOGGER_LEVEL, TT_LOGGER_TYPES, TT_METAL_LOGGER_FILE, TT_METAL_LOGGER_LEVEL, TT_METAL_LOGGER_TYPES, TT_METAL_HOME, TT_METAL_THREADCOUNT, TT_METAL_CCACHE_KERNEL_SUPPORT, TT_METAL_CQ_SIZE_OVERRIDE, TT_METAL_KERNEL_READBACK_ENABLE, TT_METAL_PROFILER_DIR, TT_METAL_JIT_ANALYTICS, TT_METAL_RECORD_NOC_TRANSFER_DATA, TT_METAL_RISCV_DEBUG_INFO, TT_METAL_WATCHER_DEBUG_DELAY, TT_VISIBLE_DEVICES, TT_MESH_ID, TT_MESH_HOST_RANK, TT_SIMULATOR_LOCALHOST, NNG_SOCKET_ADDR, NNG_SOCKET_LOCAL_PORT, TT_BACKEND_CPUSET_ALLOCATOR_* (three), CI, CONTINUOUS_INTEGRATION, GITHUB_ACTIONS. Which of these a run actually asks for: PLACEHOLDER (the batch's getenv probe).
- The JIT (tt_metal/jit_build/build.cpp): a failed compile or link throws after reading its log file (:53-60), the log being `<object>.log` or `<elf>.log` in the cache (:483-486, :564-566), so the jit-stage diagnostic is in the exception text; the cache root's other entries are removed at start (:72; PHASE-NOTES P4); the default cache root is under HOME, or /tmp without HOME (:90, :92; PHASE-NOTES P4 cites :86-93); TT_METAL_CCACHE_KERNEL_SUPPORT makes the JIT use ccache (:130).
- Thread pool: tt_metal/common/executor.hpp:12-13 sizes the executor from TT_METAL_THREADCOUNT when set; its default: PLACEHOLDER (local/source-excerpts.txt). If the default is the host's CPU count, a run on alpha01's 256 CPUs meets the sandbox's TasksMax of 256, as OpenMP did (PHASE-NOTES P1, multicore proxy); the batch's sandbox attempt measures it.
- UMD (tt_metal/third_party/umd/device): its robust mutexes are named shared-memory objects created with shm_open, prefix TT_UMD_LOCK. (utils/robust_mutex.cpp:51, :258-260), which LocalChip creates for PCIe chips (chip/local_chip.cpp:119, :134); warm reset names a listener directory on the root filesystem, /tmp/tt_umd_listeners (api/umd/device/warm_reset.hpp:63, warm_reset.cpp); a simulator path ending in .so loads the library in process and takes soc_descriptor.yaml from its directory (simulation/simulation_chip.cpp:23, :32-33), while TT_SIMULATOR_LOCALHOST and the NNG variables belong to the socket-based simulator host (simulation/simulation_host.cpp:56). Whether a ttsim run touches /dev/shm or /tmp: PLACEHOLDER (the batch runs every measuring step with private /tmp, /var/tmp, and /dev/shm and lists what each left there).

### ttsim v1.3.4 (libttsim_wh.so strings)

- 813 strings of 6 or more characters (exec e37d). The words UndefinedBehavior, UnimplementedFunctionality, and UnsupportedFunctionality each appear once, beside the format `[%ld] ERROR: %s: %s%s`, so a finding is presumably printed as `[<n>] ERROR: <class>: <message>` (confirmed or corrected by the captures: PLACEHOLDER). A symbol decode_and_execute_unimplemented is present.
- Message strings: "Wormhole does not support Zicsr"; "fence.i does not flush the instruction cache on babyrisc and should not be used"; "fence instructions do not enforce memory ordering on Wormhole and should not be used"; "could not decode instruction inst=0x%x at pc=0x%llx"; "unaligned addr=0x%llx size=%d"; "unaligned target_pc=0x%llx"; "invalid pc=0x%llx"; "invalid x=%d"; "invalid y=%d"; "TLB region overrun: offset=0x%x size=%d"; "bar0 overrun ..."; "bar0: misaligned offset=0x%x"; "DRAM mmap() failed"; "SrcA bank is not valid" and its SrcB and already-valid variants; "pipe %d inst fifo full"; and a speed report, "%.1f seconds (%.1f MHz)" or "(%.1f KHz)" [... trimmed ...]. Which message carries which class is not in the strings; the seeded captures measure it.
- No string names getenv or a TTSIM variable, so the library reads no environment variable of its own (inferred from the strings; not measured). It exports libttsim_init, libttsim_exit, libttsim_clock, libttsim_pci_config_rd32/wr32, libttsim_pci_mem_rd_bytes/wr_bytes, libttsim_pci_dma_mem_rd_bytes/wr_bytes, libttsim_set_pci_dma_mem_callbacks, libttsim_tile_rd_bytes/wr_bytes, and ttsim_rv32_get_core_active/set_core_active.

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

Reading: no Tier A host program sets fp32_dest_acc_en or unpack_to_dest_mode, and every circular buffer is Float16_b, so no reference takes the Float32 unpack path of ttsim issue #18; two references have no compute kernel at all. The ComputeConfig defaults (fp32_dest_acc_en false, unpack_to_dest_mode empty, meaning Default): PLACEHOLDER (local/source-excerpts.txt). The batch reads the descriptors the JIT generated in each clean run's cache: the lines themselves go to local/unpack_to_dest.txt only (JIT-generated text), and report/unpack_to_dest-facts.txt holds counts derived from them (lines naming unpack_to_dest or dest accumulation, and how many say true). Those counts point at the lines; the lines are the evidence. Whether ttsim reports the issue's `tensix_unpacr` UndefinedBehavior: PLACEHOLDER.

### The examples' own checks (reading)

- loopback: a pass flag (loopback.cpp:30) cleared on a mismatch (:134, :141) after a size check (:127); prints "Test Passed" (:150), else throws "Test Failed" (:152).
- eltwise_binary: a size check (eltwise_binary.cpp:164) and a per-element comparison that prints "Result mismatch at index ..." and clears the flag (:170-171); "Test Passed" (:186), else "Test Failed" (:188). The comparison's tolerance: PLACEHOLDER (local/source-excerpts.txt).
- eltwise_sfpu: a per-element comparison (eltwise_sfpu.cpp:152-153); "Test Passed" (:168), else "Test Failed" (:170). Tolerance: PLACEHOLDER.
- matmul_single_core and matmul_multi_core: the PCC of the bfloat16 result against a CPU golden (check_bfloat16_vector_pcc), printed as "Metalium vs Golden -- PCC = <value>" and required to exceed 0.97 (matmul_single_core.cpp:239-241, matmul_multi_core.cpp:332-334); "Test Passed" (:253, :346).
- add_2_integers_in_riscv: no line matches pass, fail, mismatch, PCC, TT_FATAL, TT_THROW, EXIT_FAILURE, or a throw (exec b0ab), so it seems to print its result without a check of its own. What it prints: PLACEHOLDER. If so, P4.11's smoke driver ("passes its own check") needs the expected value stated, which is a consequence for P4.11, not an owner choice.

## The batch run

Files: plans/spikes/p4-ttsim-runtime/run.sh (the orchestrator), spike.py (stdlib helpers: measure, seed, quiet, strace and getenv summaries, summarize), sandbox_attempt.py (runs one example through lassi.executors.sandbox), getenv_log.c (an LD_PRELOAD shim that logs which variables a process asks for, never their values). Each file's header states its contract.

Command, from a clean commit in the detached clean worktree (plans/LESSONS.md, alpha01), with `rx doctor` and a du check before and after (P4 plan), and an explicit `--timeout` (rx job start otherwise sends 86400 s, tools/rx.py:444, which the gate clamps to 172800, so a runaway would get a day):

```
uv run tools/rx.py doctor
uv run tools/rx.py job start --big --name p49-ttsim-runtime --timeout 10800 -- 'bash plans/spikes/p4-ttsim-runtime/run.sh'
uv run tools/rx.py job wait <id> --timeout 600 --interval 300 -n 120
uv run tools/rx.py doctor
```

The batch's stdout stays under 120 lines (worst case about 100: a header, one line per step, the findings, and the checks); `-n 120` shows the whole stdout. The step table goes to report/summary.txt only.

Stopping it: do not use `rx job kill`. It kills the runner and frees the slot (tools/server/gate.py:752-764) while bash is still reading run.sh from that slot; the batch may then die at its next write to the job's stdout, skipping the checks and the summary. run.sh sets `trap '' PIPE`, but that does not keep the batch alive after a runner kill (a write still fails, and the slot is already free), so the stop file is the only safe stop. Instead touch the per-run stop file, which the batch checks between steps and (through spike.py measure --stop-file) during a step: `uv run tools/rx.py exec -- 'touch /mnt/nvme10/joseph_ufl/lassi-runs/p49-ttsim-runtime/<rx id>/STOP'`. It skips the remaining steps and still writes the summary and the checks. The per-run file is inside the run directory, so it never blocks a later batch and no agent has to delete it. The owner's gate STOP ($LASSI_SCRATCH/lassi-gate/STOP) is honored too and refuses a fresh start; a shared $LASSI_RUNS_ROOT/p49-ttsim-runtime/STOP is honored only while it is fresh (newer than this batch's start), so a stale one does not block later batches. The longest a step can run after a stop file appears is its own limit: up to 1830 s for the probe, 1230 s for watcher-add2, or wall+120 s (up to 1020 s) for a sandbox step, since spike.py measure polls the stop files each second and stops the group.

Output under $LASSI_RUNS_ROOT/p49-ttsim-runtime/<rx id>/:

- report/: text evidence, ASCII only; pulled into results/p4-ttsim-runtime/.
- fixtures/: every seeded and Watcher capture, ASCII only; pulled into the untracked .rx/pulls/ and copied by hand (below).
- local/: capped logs, the JIT-generated descriptor lines, and source excerpts, which quote upstream text; pulled only into .rx/pulls/.
- raw/: JIT caches, working directories, seeded kernel copies, the strace log; never pulled.

The pulls run from the checkout that will commit results/. Before each `--path` pull, .rx/pulls/report, fixtures, and local must be absent (rx extracts into .rx/pulls/<basename> and `--into` copies everything there, so a leftover from an earlier pull would be copied in); the operator removes only that local cache, nothing on the host.

```
uv run tools/rx.py pull <id> --into results/p4-ttsim-runtime
uv run tools/rx.py pull --path lassi-runs/p49-ttsim-runtime/<id>/report --into results/p4-ttsim-runtime
uv run tools/rx.py pull --path lassi-runs/p49-ttsim-runtime/<id>/fixtures
uv run tools/rx.py pull --path lassi-runs/p49-ttsim-runtime/<id>/local
```

Then write results/p4-ttsim-runtime/summary.md, citing provenance.json (AGENTS.md, Results; LESSONS): the device, the pins, the confirmed settings, and the acceptance results, every wall time labelled simulator. P4.9's Files line in the plan lists the spike doc, tests/executors/fixtures/, and docs/BIBLE.md; results/p4-ttsim-runtime/ is the extra path this run adds.

The last two pulls land in .rx/pulls/fixtures and .rx/pulls/local. Only these captures are copied into tests/executors/fixtures/ttsim/, each with its stdout.txt, stderr.txt, and status.json, plus a README that states the device (ttsim v1.3.4 with tt-metal 5280a9cf, simulator), that every time or rate inside stdout.txt, stderr.txt, and watcher.log is simulator wall time (exploratory, never performance), and the commit, rx id, dirty flag, pins, and each seeded line (status.json carries commit, rx_run_id, dirty, snapshot_of, and upstream_lines_removed):

- One UndefinedBehavior capture (the first seed ttsim classes so), for P4.6's reading of UB as a failed run fed back to the model.
- One gap capture per gap class reached (UnimplementedFunctionality, UnsupportedFunctionality), for the sim-gap end reason.
- seed-jit-error, for the jit-stage diagnostic.
- seed-hang, for the hang diagnostic.
- watcher-hang, only if Watcher works, for the dump P4.11 adds to a hang.

The rest (a second capture of a class, the unbuffered control seed, watcher-add2) stays in .rx/pulls/. write_fixture already replaces every upstream-derived line (a compiler source-context line other than the seeded one, a compile-command, backtrace, or TT_FATAL/TT_ASSERT/TT_THROW line) with one fixed marker, deterministically in code, so the tracked fixture is never hand-edited; status.json counts the replacements. report/fixture-check.txt then reports, per capture, that no source-context line remains (a count and line numbers, never the lines; any quoted line goes to local/fixture-context.txt).

Steps, in order. The acceptance items (the clean runs, the probe, the seeds, Watcher, the sandbox) run first; the extras (sfplm, the unbuffered control) run last, so a tight budget drops an extra, never an acceptance item. Each step has its own wall limit: spike.py measure stops the step's process group at the limit, and an inner `timeout -k 10` 30 s later stops it even if the helper has died. A step is skipped when the 9000 s budget cannot hold its limit, when a stop file in force says so, when the run directory passes its 6 GiB cap (du before each step), or when the scratch root re-measured before the step plus the run directory would pass the P4 plan's 115 GiB stop line.

1. Clean runs of the six examples with the bible's settings (TT_METAL_SIMULATOR, the descriptor beside it, TT_METAL_SLOW_DISPATCH_MODE=1, TT_METAL_DISABLE_SFPLOADMACRO=1), plus TT_METAL_RUNTIME_ROOT, TT_METAL_CACHE and TT_METAL_LOGS_PATH inside the step, a HOME of the step's own, LANG=C with LC_ALL=C, and TT_METAL_INSPECTOR_RPC=0; 900 s each; /usr/bin/time -v for peak memory; a once-a-second sampler for peak tasks. If the gate's example fails fast with loopback down, it runs once more with loopback up inside its own network namespace (kind clean-lo-up), and later steps follow that run's result.
2. The probe: the gate's example under strace (file, network, and exec calls) with the getenv shim and no HOME; one rerun with HOME if it fails fast, so a HOME dependency does not lose the probe's answer. Its network summary shows any socket call, TCP or unix, that failed.
3. Seeded failures on copies of the gate example's kernel, one inserted line each (spike.py SEEDS), found through the step's working directory (kernel.cpp:64-68 checks the cwd first) and also named by TT_METAL_KERNEL_PATH: three UB candidates (an unaligned 4-byte load, `fence`, `fence.i`), four gap candidates (a CSR read, an undecodable word, `ecall`, `wfi`, so one job captures a gap class whichever way ttsim classes each), a call to an undeclared function (the JIT error), and a wait on an L1 word nothing writes (the hang). Every seed runs line-buffered (stdbuf -oL -eL), so a printed finding is not lost if the step is killed; one unbuffered control (in the extras) answers the buffering question. step_record scans each step's own JIT cache for the seeded path or SEED_MARK and records seeded_copy_in_cache, direct proof the copy was compiled.
4. Watcher: the gate's example with TT_METAL_WATCHER=1, then the seeded hang with it; the hang's limit is sized from the measured watcher-add2 run, not the non-Watcher hang, so Watcher's own overhead does not stop it before its dump.
5. The sandbox: the gate's example through lassi.executors.sandbox (SandboxSpec with $LASSI_SCRATCH, $HOME, and the runs root hidden, $LASSI_TOOLCHAINS read-only, the workdir under the run, 2048 MiB disk cap, TasksMax 256), wall capped at 900 s so it cannot starve the extras, memory twice its peak resident set plus 4 GiB (8 to 64 GiB); a one-shot HOME retry on a fast failure, then a TT_METAL_THREADCOUNT=16 retry; then eltwise_binary, the one compute example checked in the sandbox. At this commit ENVIRONMENT_NAMES holds no TT_METAL_* name, so the example gets its variables through its argv (`env -i NAME=value... <example>`, with LANG=C and LC_ALL=C), and the attempt records SandboxSpec's refusal of the same variables as SandboxSpec.environment.
6. Extras (not acceptance): eltwise_sfpu without TT_METAL_DISABLE_SFPLOADMACRO (plans/spikes/p4-tt-pins.md, consequences), and one unbuffered control seed.

The measuring runs (steps 1, 2, 3, 4, 6) run outside the sandbox. The seeded lines are hand-written test inputs, not model output, run only on ttsim's simulated cores under an unmodified upstream host program, so they stay within the plan's outside-the-sandbox allowance and Agent Rule 6; the batch replaces the plan's "several rx run calls" with one big job. Each measuring run is confined by:

- `unshare -rmnipf --kill-child --mount-proc`: its own user, mount, network, IPC, and pid namespaces. The network namespace means no run reaches the host's network or its localhost ports (TurboQuant's tt-metal may listen there, e.g. Inspector's localhost:50051); the IPC namespace means no SysV IPC or message queue is shared with the account's other runs; the pid namespace with --kill-child means the whole process tree, even a child that calls setsid, dies with the namespace when the step ends. Loopback stays down, as in the sandbox, except in the one-shot fallback of step 1, inside the run's own namespace. The host network is never used.
- Private tmpfs mounts on /tmp, /var/tmp, /dev/shm, and /run (as the P0.16 sandbox privatizes /run), listed before they go away (raw/<step>/private-tmp.txt).
- Read-only recursive binds (--rbind, the form the sandbox measured on alpha01) of $LASSI_TOOLCHAINS (both pinned trees), the TurboQuant checkout, and the shared default JIT cache root ($HOME/.cache/tt-metal-cache, whose entries the JIT would otherwise prune), made read-only with mount_setattr. A failed mount stops the step before its program runs, and a `.ready` marker records that the wrapper reached the program, so a mount failure is not read as the program's exit status. In the local stub dry run every attempted write into a pinned tree was refused.
- `env -i` for the whole environment, with LANG=C and LC_ALL=C; the wrapper chain itself also runs under LC_ALL=C. The gate's locale is en_US.UTF-8, under which GCC diagnostics carry UTF-8 quotes (bible, Host Facts), and the kernel compiler is a GCC. P4.11 must set the same locale. As a second guard, spike.py writes every file under report/ and fixtures/ as ASCII (any other character a backslash escape) and, after the checks, rewrites any file there that still holds a non-ASCII byte.
- prlimit --core=1 (one byte, verified below), --fsize (a 2 GiB per-file cap), and --as (a 1024 GiB per-process address-space backstop; glibc's per-thread malloc arenas on 256 CPUs can reserve about 128 GiB of address space, so a lower cap could fail every step) on the whole chain; spike.py measure also stops a step whose stdout.txt plus stderr.txt pass 2 GiB. No memory or task cgroup is used: RLIMIT_NPROC is unsafe under the shared uid, systemd-run --user would add a user-manager dependency the measuring path does not otherwise need, and the pid namespace with --kill-child already bounds the process tree.
- Its own JIT cache, so no step can reuse another's kernel binary and the JIT's cache clearing sees only the step's own root.

The core limit is set at the top with `prlimit --pid $$ --core=1:1` (bash's `ulimit -c 1` is 1024 bytes, which does not disable a piped systemd-coredump), and preflight refuses to start unless getrlimit(RLIMIT_CORE) is exactly (1, 1). So a crash (the jit-error seed makes tt-metal throw, which ends in SIGABRT) stores no root-owned core under /var/lib/systemd/coredump on the root filesystem (Agent Rule 7; the OQ-014 incident). In the probe, strace's first child is a shell that always exits normally (`/bin/sh -c '"$@"; exit "$?"' sh`): when strace's first child dies of a signal, strace sets its own RLIMIT_CORE to 0 and re-raises the signal (strace.c terminate()), and a limit of 0 does not stop a piped core on alpha01 (P0 probe G1). The local stub dry run confirmed RLIMIT_CORE (1, 1) inside every measuring step and no core file after a real abort of the program in a non-probe step, and it checked the probe path with a strace stand-in that re-raises its child's signal after dropping its own limit to 0 (exploratory: the stand-in mimics strace's behavior, it is not strace; WSL's core handling is not alpha01's).

Checks after the steps (report/checks.txt): counts only, no raw path. Whether this account changed anything under /tmp, /var/tmp, or /dev/shm (a count; the path list goes to local/root-fs-changed.txt, since it filters by account, which TurboQuant shares, so a name could be another project's); any path changed in the pinned trees (one find into local/pinned-changed.txt); top entries changed in the default cache roots; and du of the scratch root before and after and the run directory against its cap. What can be attributed to a step is its own private listing, raw/<step>/private-tmp.txt, summarized in report/files-written.txt. The summary runs before the checks, so if the gate's timeout fired during the checks the fixtures and tables would still exist.

The uv warm-up before the sandbox steps builds the slot's .venv and the uv cache on the scratch disk (UV_CACHE_DIR, which the gate sets), under `timeout -k 30 900` and `uv run --frozen --offline`, so a missing locked package fails here and never fetches from the host. It runs with PYTHONDONTWRITEBYTECODE=1, so the import leaves no bytecode in the slot.

Every record that holds a wall time names the device (ttsim v1.3.4 on the host CPU) and says it is simulator wall time: the first line of report/runs.tsv, the header of report/summary.txt, the batch.txt adaptive-limits line, and the device and timing_note fields of report/steps/*.json and of each capture's status.json. status.json also carries the commit, the rx run id, dirty, snapshot_of, and the step's start and end times.

Local stub dry run (exploratory; 2026-09-25 and 2026-09-26; a local Linux environment with stand-ins for the examples, strace, and uv, and a fake scratch root under a user namespace; no remote activity). All 24 steps ran; every path was exercised: the loopback fallback, the more numerous seeds, the hangs stopped at their limits, a lingering background process killed, the sandbox helper's unavailable path and the allowlist refusal, the read-only binds (every write into a pinned tree refused), the stop file, the ASCII rewrite, the deterministic fixture sanitizing, and the summary; RLIMIT_CORE read (1, 1) in every step, and neither a real abort in a non-probe step nor an abort under the re-raising strace stand-in in the probe left a core file; report/ held no non-ASCII byte and no /tmp path. The getenv shim was checked separately against a threaded C++ program (static initializers, threads, secure_getenv). The dry run tests the script's logic, not tt-metal or ttsim.

PROJECTED, not measured: 30 to 90 minutes and under 2 GiB of scratch; under 3 hours in the worst case (the budget, then the checks). Hence a job rather than `rx run`.

What the run can answer: for the gate's example, every acceptance item; for the five Tier A examples, exit status, own check, memory, JIT cache, and unpack_to_dest; whether the bible's settings suffice; whether the sandbox as committed runs the gate's example and eltwise_binary (the one compute example checked there); what ttsim prints for the seeded conditions; whether Watcher runs, logs, and shows a hung core's state.

What it cannot answer: behavior on silicon (Agent Rule 2; every ttsim pass stays provisional); performance (simulator wall times size limits only); the reads, writes, and environment needs of a compute-kernel example (the file, network, and environment probe runs only add_2_integers_in_riscv, which has no compute kernel; the eltwise_binary sandbox attempt is the only compute-example check); whether a seed reaches a class no seed hits (each seed tests one instruction; several gap candidates make a miss unlikely); whether the sandbox works once P4.11 adds the TT variables to ENVIRONMENT_NAMES (the attempt passes them through argv instead; the difference is the variables' route, not their values); Tier A items that P4.13 writes itself (its host programs are new); concurrency between runs (steps run one at a time); whether a run needs more than a loopback interface (the host network is never offered; the probe's network summary shows what a run tried); and behavior with the pinned trees writable (every measuring run sees them read-only, so a run that would write there fails and the probe shows the refused write).

## Results

All PLACEHOLDER until the batch runs; each value will cite report/runs.tsv or report/steps/<step>.json.

### Clean runs (simulator wall time, exploratory, sizing only)

| Example | Exit | Own check | ttsim messages | Peak RSS (MiB) | Peak tasks | JIT cache (KiB / files) | Wall (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| add_2_integers_in_riscv | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |
| loopback | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |
| eltwise_binary | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |
| eltwise_sfpu | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |
| matmul_single_core | PLACEHOLDER | PLACEHOLDER (PCC PLACEHOLDER) | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |
| matmul_multi_core | PLACEHOLDER | PLACEHOLDER (PCC PLACEHOLDER) | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |

### What tt-metal reads, writes, and needs at run time

- Reads (report/strace-summary.txt): PLACEHOLDER (the pinned tree by directory, the ttsim directory, system paths, anything else).
- Writes outside the step's cache and logs directory (report/strace-summary.txt, report/files-written.txt): PLACEHOLDER (expected: /dev/null only; the private /tmp and /dev/shm listings: PLACEHOLDER).
- Executables started: PLACEHOLDER (expected: the sfpi compiler under runtime/sfpi and the shell).
- Network calls: PLACEHOLDER.
- Environment (report/getenv.txt): variables asked for, and whether HOME is needed: PLACEHOLDER.
- The pinned trees unchanged, and the default cache roots untouched (report/checks.txt): PLACEHOLDER.

### The P0.16 sandbox, pinned tree read-only

| Attempt | Exit | Hang | Killed | program_s (simulator wall time, sizing only) | Workdir complete | Note |
| --- | --- | --- | --- | --- | --- | --- |
| sandbox-add2 | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |
| sandbox-add2-threads16 (only if the first fails) | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |
| sandbox-eltwise_binary | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |

SandboxSpec.environment with the TT variables: PLACEHOLDER (the local dry run showed the expected refusal: "the sandbox environment may not set 'TT_METAL_RUNTIME_ROOT'"; the host run confirms).

### Seeded captures (fixtures)

| Seed | Class ttsim gave | First line | Exit | Fixture |
| --- | --- | --- | --- | --- |
| ub-unaligned | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |
| ub-fence | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |
| ub-fencei | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |
| gap-zicsr | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |
| gap-decode | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |
| gap-ecall | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |
| gap-wfi | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |
| jit-error | JIT error (expected) | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |
| hang | none (timeout expected) | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |

Fixtures chosen for tests/executors/fixtures/ (one UndefinedBehavior, one per gap class reached, seed-jit-error, seed-hang, and watcher-hang if Watcher works), each sanitized by write_fixture and with a README as the procedure above states: PLACEHOLDER. Whether stdio buffering loses a finding line (the unbuffered control seed against the line-buffered ones): PLACEHOLDER.

### unpack_to_dest (from the run)

Per clean run, any_unpack_to_dest_enabled (report/unpack_to_dest-facts.txt; the descriptor lines behind it are in local/unpack_to_dest.txt, quoted upstream text) and ttsim's `tensix_unpacr` UndefinedBehavior: PLACEHOLDER.

### TT_METAL_DISABLE_SFPLOADMACRO

eltwise_sfpu without it: PLACEHOLDER (exit, check, ttsim messages, compared with the clean run).

### Watcher

| Step | Exit | watcher.log | Hung core's state in the log | Wall (s, simulator) |
| --- | --- | --- | --- | --- |
| watcher-add2 | PLACEHOLDER | PLACEHOLDER | n/a | PLACEHOLDER |
| watcher-hang | PLACEHOLDER (timeout expected) | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |

### Host checks

Root filesystem (/tmp, /var/tmp, /dev/shm; filtered by account, which TurboQuant shares, so read beside each step's private listing): PLACEHOLDER. Pinned trees: PLACEHOLDER. Default cache roots (changes by anyone): PLACEHOLDER. Scratch du before and after: PLACEHOLDER. Run directory size against its 6 GiB cap: PLACEHOLDER.

## Settings the bible's ttsim row lacks

From the reading; each is confirmed or dropped by the run.

1. TT_METAL_RUNTIME_ROOT on the pinned tree and TT_METAL_CACHE inside the run: already required by Toolchain Pins (Install) and PHASE-NOTES P4, but absent from the row.
2. TT_METAL_LOGS_PATH inside the run: Inspector, on by default, writes under `<logs dir>/generated/inspector`, and Watcher and DPRINT under `<logs dir>/generated/`. Default logs directory: PLACEHOLDER; if it is the runtime root, a run without the variable writes into the pinned tree, which the sandbox mounts read-only.
3. TT_METAL_INSPECTOR_RPC=0, or TT_METAL_INSPECTOR=0: Inspector's RPC server listens on localhost:50051 by default. Which of the two, and whether Inspector's own files are worth keeping: PLACEHOLDER (report/files-written.txt shows what Inspector wrote).
4. TT_METAL_THREADCOUNT: PLACEHOLDER (needed only if the default pool passes the sandbox's TasksMax).
5. HOME: PLACEHOLDER (the probe runs without it).
6. Single chip: ttsim presents one Wormhole chip; how tt-metal reports it: PLACEHOLDER. No variable is needed unless the log says otherwise.
7. TT_METAL_DISABLE_SFPLOADMACRO: PLACEHOLDER (whether it changes any result at the pin).
8. LANG=C and LC_ALL=C: the kernel compiler is a GCC, whose diagnostics carry UTF-8 quotes under the gate's en_US.UTF-8 locale (bible, Host Facts), so jit-stage diagnostics would break the ASCII rule. The batch sets both; the compile sandbox already does (bible, Sandbox).

## Finding

PLACEHOLDER (two sentences, written from the report after the run).

Confidence: PLACEHOLDER.

## Consequences for the plan

Firm now, from the reading:

- P4.10 and P4.11: the kernel search looks in the working directory first (kernel.cpp), so kernel sources placed at their relative path in the workdir are found, and the executor must keep any other tt_metal/ tree out of a workdir.
- P4.11: under the simulator, tt-metal waits for cores without a timeout (llrt.cpp:311-320), so a hang ends only at the executor's wall limit; the Harness Contract's timeout rule stands, and the hang diagnostic must come from the executor, not from tt-metal.
- P4.11: the sandbox allowlist gains the TT variables the row names (the P4 plan already says so); TT_METAL_LOGS_PATH and an Inspector setting join them if the run confirms items 2 and 3 above.
- P4.6: the parser matches ttsim's `ERROR: <class>: <message>` lines (format from the library strings; the captures confirm).
- P4.11: the ttsim program's environment sets LANG=C and LC_ALL=C, as the batch does and as compiles already do, so jit-stage diagnostics stay ASCII.
- P4.11: a hang on ttsim runs until the wall limit, and `rx job kill` stops only the gate's runner, so any long batch that runs ttsim programs needs its own stop mechanism, as this one has.

After the run: PLACEHOLDER (P4.6 fixtures, P4.11 limits from peak memory, tasks, and JIT cache size, the Watcher decision, P4.12's cache reading, P4.13's tolerances and unpack_to_dest notes, P4.G's smoke check for add_2_integers_in_riscv).

## Proposed bible edit

Drafted after the run, from report/, and applied to the mirror and the master copy by the task that holds docs/BIBLE.md at the time (AGENTS.md, Authority). Outline, every value PLACEHOLDER:

- Execution Backends, ttsim row, Key settings: add TT_METAL_RUNTIME_ROOT, TT_METAL_CACHE and TT_METAL_LOGS_PATH in the workdir, LANG=C with LC_ALL=C, and the Inspector and thread-count settings the run confirms. Leave Status at Planned (the registered ttsim Executor is P4.11; in that table Status means the executor is available, as the native row shows, so flipping it in P4.9 would claim more than the code does). Note in ttsim Facts, not the Status column, that the pinned examples ran on ttsim [MEASURED <date>, rx id, simulator].
- ttsim Facts: the finding format and the class of each seeded condition; the six examples' results at the pin (simulator, provisional until silicon); the unpack_to_dest check of each Tier A reference; the infinite wait under the simulator; what a run reads, writes, and needs.
- Harness Contract, timeout bullet: replace "if Watcher works under ttsim ([OPEN])" with the finding.
- Risks And Questions, question 4: answered, with the date and this spike.
- Decision Log: one new top row for P4.9, and the count sentence raised by one (re-read the current count in docs/BIBLE.md when the edit is applied; do not quote a stale number here).

## Owner queue

No item is expected: the question is factual. Two may follow the run:

- If Watcher works but slows ttsim runs substantially, whether P4.11 runs every attempt with Watcher, or reruns a hung attempt with it, trades wall time on the shared host against a dump on the first hang. Draft once the cost is measured: PLACEHOLDER.
- raw/ (the JIT caches and working directories, possibly a few GiB) stays on the host under the 120G cap, since agents never delete there. Once the report and fixtures are pulled and confirmed, recommend to the owner that raw/ under $LASSI_RUNS_ROOT/p49-ttsim-runtime/<rx id> be backed up (zipped to the workstation) or removed. The report and fixtures hold everything the spike needs.

## Sources

Read on 2026-09-25.

- alpha01: rx doctor (about 18:34, UTC-07:00); rx 20260925-183429-exec-b0ab, 20260925-183618-exec-1845, 20260925-183803-exec-e37d. Batch: PLACEHOLDER.
- The pinned tree, $LASSI_TOOLCHAINS/tt-metal@5280a9cf (tt-metal 5280a9cfb00998fd49667a29523d03aee905c129; results/p4-tt-install/summary.md): tt_metal/llrt/rtoptions.cpp, rtoptions.hpp, llrt.cpp, tt_cluster.cpp, core_descriptor.cpp; tt_metal/impl/kernels/kernel.cpp; tt_metal/impl/context/metal_context.cpp; tt_metal/impl/debug/watcher_server.cpp, dprint_server.cpp, noc_logging.cpp, inspector/data.cpp; tt_metal/jit_build/build.cpp, genfiles.cpp; tt_metal/common/executor.hpp; tt_metal/api/tt-metalium/kernel_types.hpp; tt_metal/programming_examples/CMakeLists.txt and the six examples' host programs and kernels; tt_metal/third_party/umd/device/utils/robust_mutex.cpp, chip/local_chip.cpp, api/umd/device/warm_reset.hpp, warm_reset.cpp, simulation/simulation_chip.cpp, simulation/simulation_host.cpp.
- $LASSI_TOOLCHAINS/ttsim@v1.3.4/libttsim_wh.so (sha256 7b10aa05a5297c4a28f274e39526bfe6c69373661d1b4d4e47c4257e44b79507): its strings.
- plans/spikes/p4-tt-pins.md (issue #18, the CI facts at the pin), plans/spikes/p0-sandbox-hardening.md and lassi/executors/sandbox.py (the sandbox), plans/PHASE-NOTES.md (P4: the JIT cache rule).
- ttsim issue #18: https://github.com/tenstorrent/ttsim/issues/18 (as read for P4.1 on 2026-09-24; not reread here).
