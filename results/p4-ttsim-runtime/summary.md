# P4.9: ttsim runtime facts, unpack_to_dest, and Watcher

[MEASURED] provenance.json (rx job 20260926-140947-p49-ttsim-runtime-6f4c):

- commit: 755c739 (755c739e3d44a5a94c5f942f8f3872cc1fbb5fd3), clean tree (dirty false, snapshot_of null; batch.txt: "dirty=false, changed paths 0"), run in the detached clean worktree slot desktop-8r113ei-detached-755c739e;
- host: alpha01, Ubuntu 22.04.5 LTS, kernel 6.6.29+main+3.0.0r1-amd64-gio-epilmore-dev+, nproc 256; start 2026-09-26T14:09:48-07:00, end 14:15:03, 315 s for the whole job; job rc 0; 24 of 24 steps ran (summary.txt);
- toolchain pins: toolchains/tt-metal.pin (tt-metal 5280a9cfb00998fd49667a29523d03aee905c129) and toolchains/ttsim.pin (ttsim v1.3.4); their text is in provenance.json under pins, and batch.txt re-hashed libttsim_wh.so at run time: sha256 7b10aa05a5297c4a28f274e39526bfe6c69373661d1b4d4e47c4257e44b79507, the pinned value;
- device: ttsim v1.3.4 (libttsim_wh.so, a virtual Wormhole) on alpha01's CPU, with tt-metal 5280a9cf: a simulator, not silicon. Every wall time below is simulator wall time and every rate line the simulator's own clock rate: exploratory, sizing only, never performance. Every pass is provisional until confirmed on silicon.

Command: `bash plans/spikes/p4-ttsim-runtime/run.sh` as an rx job with --timeout 10800 (provenance.json, cmd). plans/spikes/p4-ttsim-runtime.md gives the design, the findings, and its reading of the full logs and generated files, which are kept locally under .rx/pulls/ and not in the repository.

Each measuring run (batch.txt): env -i with PATH=/usr/sbin:/usr/bin:/sbin:/bin, LANG=C, LC_ALL=C, TMPDIR=/tmp, TT_METAL_RUNTIME_ROOT on the pinned tree, TT_METAL_SIMULATOR on the pinned libttsim_wh.so, TT_METAL_SLOW_DISPATCH_MODE=1, TT_METAL_INSPECTOR_RPC=0, plus TT_METAL_DISABLE_SFPLOADMACRO=1 (not in sfplm-on), TT_METAL_CACHE and TT_METAL_LOGS_PATH in the step, and HOME in the step (the probe: none); unshare -rmnipf --kill-child --mount-proc, private tmpfs on /tmp, /var/tmp, /dev/shm, and /run, the toolchains root read-only; prlimit --core=1 (one byte, verified by getrlimit), --fsize=2147483648, --as=1099511627776. Limits (batch.txt): clean 900 s each, seed and hang 120 s.

## Results

Clean runs (runs.tsv; wall_s is simulator wall time; maxrss_kib is GNU time's; peak_tasks is the process group's):

| Step | rc | check | pcc | wall_s | maxrss_kib | peak_tasks | cache KiB / files | sim_rate_line |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean-add_2_integers_in_riscv | 0 | none printed (a) | | 1.39 | 100352 | 267 | 2996 / 70 | 0.4 seconds (12.1 KHz) |
| clean-loopback | 0 | passed | | 1.42 | 102400 | 269 | 3100 / 70 | 0.5 seconds (15.8 KHz) |
| clean-eltwise_binary | 0 | passed | | 1.64 | 116736 | 273 | 6088 / 120 | 0.5 seconds (21.1 KHz) |
| clean-eltwise_sfpu | 0 | passed | | 1.49 | 110592 | 296 | 6260 / 120 | 0.5 seconds (19.5 KHz) |
| clean-matmul_single_core | 0 | passed | 0.9806516 | 3.73 | 106496 | 276 | 5980 / 120 | 2.7 seconds (135.7 KHz) |
| clean-matmul_multi_core | 0 | passed | 0.99999344 | 2.9 | 106496 | 285 | 6036 / 120 | 1.9 seconds (7.8 KHz) |

(a) The batch's check pattern knows "Test Passed" and the PCC line, not this example's own success line; the spike report reads that line from the full log. No clean run reported a ttsim error class (runs.tsv sim_classes). Each clean run's program peaked at 257 threads (runs.tsv peak_process_threads); among the other steps, seed-jit-error peaked at 144 and the two Watcher runs at 258.

Seeded runs on copies of the gate example's kernel, one inserted line each (runs.tsv, summary.txt, steps/seed-*.json):

- seed-ub-unaligned (a 4-byte load from an address 2 mod 4): rc 1; sim_lines `[4500] ERROR: NonContractualBehavior: rv32_mem_rd: unaligned addr=0x16dfe2 size=4` (sim_classes is empty because the batch's class pattern lists only three classes).
- seed-ub-fence: rc 1, UnsupportedFunctionality: decode_and_execute_fence: fence instructions do not enforce memory ordering on Wormhole and should not be used.
- seed-ub-fencei: rc 1, UnsupportedFunctionality: decode_and_execute_fence_i: fence.i does not flush the instruction cache on babyrisc and should not be used.
- seed-gap-zicsr: rc 1, UndefinedBehavior: decode_and_execute_csrrs: Wormhole does not support Zicsr.
- seed-gap-decode: rc 1, UndefinedBehavior: decode_and_execute_unimplemented: could not decode instruction inst=0xffffffff at pc=0x8238.
- seed-gap-ecall: rc 1, UnsupportedFunctionality: decode_and_execute_ecall_ebreak: inst=0x73.
- seed-gap-wfi: rc 1, UnsupportedFunctionality: decode_and_execute_ecall_ebreak: inst=0x10500073.
- seed-jit-error: rc 134; jit_first is tt-metal's critical log line reporting that the brisc build failed.
- seed-hang: rc 137, timed_out True, stopped_reason limit, wall_s 131.0 against its 120 s limit.
- seed-ub-unaligned-unbuffered (extra, no line buffering): rc 1.
- No step reported UnimplementedFunctionality. seeded_copy_in_cache is True in every seeded step but seed-hang (empty).

Watcher (summary.txt, steps/watcher-*.json):

- watcher-add2 (TT_METAL_WATCHER=1): rc 0; watcher.log 26526 bytes, 222 lines; wall_s 2.98 (clean run 1.39).
- watcher-hang: rc 137, timed_out True; watcher.log 1637305 bytes, 12945 lines; wall_s 131.06. The tracked capped copy, tests/executors/fixtures/ttsim/watcher-hang/watcher.log, shows worker core (x=0,y=0) with BRISC at waypoint R in dump 1 and NSW in dumps 2, 3, 127, and 128 (it keeps dumps 2, 127, and 128 whole and cuts dump 3 at its line 270), with k_ids 1, which the log maps to the example's kernel reader_writer_add_in_riscv.cpp; the last dump is #128.

The P0.16 sandbox (summary.txt, steps/sandbox-*.json): Limits cpus 16, memory_mb 8192, disk_mb 2048, wall_s 60, tasks_max 256, the toolchains root read-only; hang False, killed False, and workdir_incomplete False in all four.

| Step | TT_METAL_THREADCOUNT | rc | check | program_s (simulator) | cache KiB / files |
| --- | --- | --- | --- | --- | --- |
| sandbox-add2 | unset | 134 | none printed | 0.09 | 0 / 0 |
| sandbox-add2-home (HOME set) | unset | 134 | none printed | 0.09 | 0 / 0 |
| sandbox-add2-threads16 | 16 | 0 | none printed (a) | 1.32 | 2996 / 70 |
| sandbox-eltwise_binary | 16 | 0 | passed | 1.42 | 6088 / 120 |

SandboxSpec.environment with the TT variables (summary.txt): "refused: the sandbox environment may not set 'TT_METAL_RUNTIME_ROOT'; allowed names: LANG, LC_ALL, NVHPC_CUDA_HOME, OMP_NUM_THREADS, PATH, TMPDIR".

TT_METAL_DISABLE_SFPLOADMACRO unset (runs.tsv): sfplm-on-eltwise_sfpu rc 0, passed, cache 6260 KiB / 120 files, as in the clean run.

The probe, the gate's example under strace with no HOME (steps/probe-add2.json, strace-summary.txt, getenv.txt): rc 0. Network calls: none. /dev: /dev/urandom only. /etc files read: /etc/ld.so.cache and /etc/localtime. Writes outside the run directory and the private directories: one, a failed mkdir of the pinned tree's generated directory. Executables: the pinned sfpi compiler (riscv-tt-elf-g++ 33 times, with GCC 15.1.0's cc1plus, lto1, lto-wrapper, collect2, as, and ld), /bin/sh (18), /usr/bin/make (16), /usr/bin/env (1), and the example. The example asked for 156 distinct names; set among them were only TT_METAL_CACHE, TT_METAL_DISABLE_SFPLOADMACRO, TT_METAL_INSPECTOR_RPC, TT_METAL_LOGS_PATH, TT_METAL_RUNTIME_ROOT, TT_METAL_SIMULATOR, and TT_METAL_SLOW_DISPATCH_MODE; no process the getenv shim logged (the example and the GCC processes) asked for HOME.

Files outside the JIT cache (files-written.txt): each step's logs directory held Inspector's YAML files under generated/inspector and, once kernels compiled, generated/watcher/kernel_names.txt and kernel_elf_paths.txt; no HOME directory received a file; the private /tmp, /var/tmp, /dev/shm, and /run listings held nothing or only /run/mount.

unpack_to_dest (unpack_to_dest-facts.txt): unpack_to_dest_lines is 0 in every run. any_unpack_to_dest_enabled reads true in every run with a descriptor, which is the batch's false positive: its pattern counts any nonzero digit, and the lines it reads carry an occurrence count (plans/spikes/p4-ttsim-runtime/spike.py, UNPACK_ENABLED and unpack_facts). The spike report reads the line itself, which sets dest accumulation off. No clean run reported UndefinedBehavior (runs.tsv).

Fixture check (fixture-check.txt): 0 compiler source-context lines and 0 compile-command lines in each of the 12 captures. Six are tracked in tests/executors/fixtures/ttsim/ (README.md there).

Host checks (checks.txt): "root-fs check: passed; this account changed nothing there since the start (59 unreadable, 0 raced)"; pinned trees: 0 paths changed in tt-metal@5280a9cf and ttsim@v1.3.4; the default cache root under the shared HOME: 0 top entries changed; /tmp/tt-metal-cache and /tmp/tt_umd_listeners absent; scratch du -sk before 101204520 KiB, after 101297648 KiB; run directory 92340 KiB against its 6291456 KiB cap. provenance.json: scratch_free_gb 582.1 before and 582.0 after, root_free_gb 302.5. A read-only listing of /var/lib/systemd/coredump after the job (rx 20260926-142038-exec-5ccd) showed no file from the job; its newest files date from 2026-09-25 (OQ-034).
