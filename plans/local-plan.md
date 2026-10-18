# Local Plan: LASSI With No Hardware And No alpha01

- Status: a proposal for the owner (J). Nothing in it is adopted. Apart from section 4.2's step 1, which needs no decision, no rule, pin, task state, or bible text changes until the owner answers the decisions in section 7.
- Date: 2026-10-15. Base: branch p5-ir at 422876c.
- Evidence: plans/spikes/local-host.md (the capacity spike, 2026-10-15; untracked when this was written), the repository at 422876c, and public package indexes read on 2026-10-15 (section 2.3). Nothing was installed, configured, or committed, no rx command ran, and no device was queried beyond the spike's read-only nvidia-smi.
- Labels: [MEASURED 2026-10-15, workstation] is a read-only probe whose command and output the spike logs, in its sections 1 to 5 or in its section "Probes of the planning round (2026-10-15)". [MEASURED ..., alpha01] cites earlier alpha01 evidence. PROJECTED is an estimate, given with its basis. (inferred) marks reasoning. (index) marks a package index read on 2026-10-15. (exploratory) marks a run of uncommitted code.

In short:
- The workstation has the capacity for P5 and for the hardware-free parts of P11, P12, and P16: 16 threads, 15.6 GiB for WSL, and 721 GiB free.
- Four things stop it today, and none of them is capacity. The sandbox setup exits 16 in WSL. No pinned host tool is installed. The install scripts accept only alpha01's root. P4's tt-metal and ttsim installs exist only on alpha01.
- A dedicated Ubuntu 22.04 WSL distro would keep every pin. The package indexes still serve the exact pinned compiler builds.
- Every accelerator phase stays BLOCKED, and no gate is restated.
- Fourteen owner decisions follow (section 7); the fourteenth only confirms OQ-004's answer. Decisions 1 to 5 gate all local building: no install, .wslconfig change, or LLVM-scale build before them. Decision 6 gates any [MEASURED] local result, and decision 7 the sandbox change.

## 1. The direction and its reading

The owner's direction of 2026-10-15 (plans/LESSONS.md, Owner directions): "Tenstorrent cards are still down. Hardware is unavailable, including Furiosa. Do all work locally for the forseeable future. Plan around hardware unavailability."

This plan reads it as follows:

- No Tenstorrent, RNGD, MI300X, or other cluster device is used. No alpha01 work of any kind happens: no rx command, ssh, scp, rsync, or sftp, until the owner says otherwise.
- The workstation (Windows plus WSL) is the only host. Builds, simulator runs, and sandboxed runs move there or wait.
- ttsim is a simulator running on the host CPU, so the Tenstorrent target stays within reach. Every TT result names ttsim v1.3.4 on tt-metal 5280a9cf as its device, never silicon (Agent Rule 2).
- The workstation's own RTX 3080 is not cluster hardware, and the direction does not say whether it may be used. That is decision 8, not part of this reading.
- Agent Rules 8, 9, and 13 have nothing to act on. Agent Rules 7 and 17 still bind, but they name only alpha01, so local builds need policy of their own (decisions 3 and 4).
- Two owner-queue items become moot: OQ-054 (P5.4's install on alpha01) and OQ-045 (the real-model Tier A run on RNGD). See decision 10.

## 2. What the workstation can do

### 2.1 Capacity [MEASURED 2026-10-15, workstation; spike sections 1 to 3 and its planning-round probes]

| Resource | Value | Bearing |
| --- | --- | --- |
| CPU | Intel Core i7-10700K: 8 cores and 16 threads, AVX2, no AVX-512 | WSL sees 16 CPUs. The gate's job formula, min(max_build_jobs, cpu_count // 2) (tools/server/gate.py:418), gives 8 here |
| Memory | 31.9 GiB RAM. WSL gets 15,951 MiB plus 4,096 MiB swap, the defaults, since .wslconfig is empty | The tight resource for LLVM-scale links (section 3.2) |
| Disk | C: 721.1 GiB free (NVMe; holds the WSL disk). Z: 578.3 (SATA SSD; the drive mapping is inferred). E: 313.9 (NVMe, 92% used). D: 387.0 and F: 1647.9 (hard disks) | Ample. P5 needs about 28 to 46 GiB (PROJECTED, section 3.2) |
| GPU | RTX 3080: 10,240 MiB, compute capability 8.6, driver 617.42. WSL reaches it through /dev/dxg and /usr/lib/wsl/lib, with no /dev/nvidia* node | Unused unless decision 8 allows it |
| WSL | WSL 3.0.3.0, kernel 6.18.54.1. Ubuntu-24.04 (24.04.2) with systemd as PID 1; a docker-desktop distro, stopped | |

### 2.2 What works and what does not [MEASURED 2026-10-15, workstation; spike sections 3 to 5 and its planning-round probes]

Everything the sandbox's mechanism needs works in Ubuntu-24.04:
- unprivileged user namespaces, including the sandbox's own `unshare -rinmpfu --mount-proc --kill-child`;
- the systemd user manager, with cpu, memory, and pids delegated;
- `systemd-run --user --scope` limits, read back as MemoryMax=64M, MemorySwapMax=0, and TasksMax=64;
- overlay and seccomp in the kernel config.

Six things do not work today:

1. **The sandbox.** The committed setup script exits 16 at `mount --move /tmp/dev /dev` (lassi/executors/sandbox.py:728) under util-linux 2.39.3, so the program never runs.
   - With -n on the four `mount --move` lines (:728, :742, :744, :747), the full composition ran: prlimit, env -i, systemd-run, and unshare (exploratory).
   - Compiles of generated sources and native and ttsim program runs go through the sandbox (bible, Sandbox).
2. **No pinned host tool is installed.**
   - WSL has gcc 13.3.0, cmake 3.28.3, ninja 1.11.1, and Python 3.12.3. It has no clang, lld, g++-12, Python 3.10, or uv.
   - tt-metal's build packages are missing too: dpkg knows none of libhwloc-dev, libnuma-dev, libtbb-dev, libcapstone-dev, libssl-dev, or pkg-config, and pkg-config is not on PATH (dpkg-query; spike, "Probes of the planning round (2026-10-15)").
3. **The install scripts accept only alpha01's root.**
   - tt-mlir.sh:40-45, tt-metal.sh:41-46, and ttsim.sh:19-24 refuse any root outside /mnt/nvme10/joseph_ufl. cuda.sh:19-20 and nvhpc.sh:13-14 refuse any root outside /mnt/nvme10.
   - Each script's check_space hard-codes alpha01's 145 GiB stop line (tt-mlir.sh:200, tt-metal.sh:180, ttsim.sh:74).
4. **P4's installs exist only on alpha01.** tt-metal@5280a9cf and ttsim@v1.3.4 were built there (results/p4-tt-install).
5. **WSL has no checkout to test from.**
   - ~/lassi-audit is a partial copy from 2026-10-11, with a Python 3.12 venv.
   - 18 test files carry the `remote` marker. They have run only on alpha01, if at all; test_tt_mlir_remote.py has never run.
6. **Core dumps land on Windows.**
   - WSL pipes cores to `|/wsl-capture-crash`, which stores them on C: under %TEMP%\wsl-crashes. A 188 MB dump from another project is there.
   - On alpha01, a 1-byte core limit stops the piped core handler (plans/spikes/p0-sandbox-hardening.md). Whether it does here is unmeasured.
   - tt-metal.sh:34 and ttsim.sh:13 still set `ulimit -c 0` (PHASE-NOTES, All Phases).

### 2.3 New fact: the pinned host tools are still served for Ubuntu 22.04 (index)

The package indexes were read on 2026-10-15 with curl. Nothing was installed.

- apt.llvm.org, jammy:
  - clang-17 and lld-17 `1:17.0.6~++20231209124227+6009708b4367-1~exp1~20231209124336.77`;
  - clang-20 and lld-20 `1:20.1.8~++20250708082409+6fb913d3e2ec-1~exp1~20250708202428.132`. This is the exact build in toolchains/tt-metal.pin:51's EXPECT_VERSION.
- packages.ubuntu.com, jammy:
  - g++-12 `12.3.0-1ubuntu1~22.04.3` (toolchains/gcc.pin:16);
  - ninja-build `1.10.1-1`;
  - python3.10 `3.10.12-1~22.04.18`;
  - util-linux `2.37.2-4ubuntu3.6`, the same upstream version as alpha01's 2.37.2.
- apt.llvm.org, noble (Ubuntu 24.04): clang-20 is a different build, `...+87f0227cb601-1~exp1~20250804210352.139`.
- CMake 3.31.6 at /usr/local/bin (tt-metal.pin and tt-mlir.pin) has no apt route to the pinned path. apt.kitware.com serves cmake `3.31.6-0kitware1ubuntu22.04.1` for jammy, but a package puts cmake in /usr/bin, not /usr/local/bin (inferred from Debian's layout; the package was not opened).

Sources:
- https://apt.llvm.org/jammy/dists/llvm-toolchain-jammy-17/main/binary-amd64/Packages
- the same path for jammy-20 and noble-20
- https://packages.ubuntu.com/jammy-updates/g++-12, /jammy/ninja-build, /jammy-updates/python3.10, /jammy-updates/util-linux
- https://apt.kitware.com/ubuntu/dists/jammy/main/binary-amd64/Packages (its cmake entry `pool/main/c/cmake/cmake_3.31.6-0kitware1ubuntu22.04.1_amd64.deb`)

## 3. P5 locally

### 3.1 Route

Each remaining P5 task splits in two:
- **A Windows half:** pure Python, unit tests with fakes, and pin and script text. It needs no host.
- **A WSL half:** installs, remote-marked tests, and ttsim and native runs.

None of the WSL half runs natively on Windows, which has no cmake, ninja, gcc, clang, or Visual Studio (spike section 3).

| Task | Windows half | WSL half | Needs first |
| --- | --- | --- | --- |
| New: local host setup | Host profile, script root and stop-line parameters, the core-limit fix, a local provenance writer, and their tests | Distro, packages, a WSL clone, uv and Python 3.10, and the remote tests as the host's acceptance | Decisions 1 to 5; decision 6 for the provenance writer and the acceptance evidence |
| New: sandbox on util-linux 2.39 | -n on four lines, the setup hash pin (tests/executors/test_sandbox.py:2636), and the bible bullet | tests/executors/test_sandbox_remote.py from a clean commit | Decision 7; a WSL clone with uv |
| New: tt-metal and ttsim rebuilt locally | Covered by host setup | The committed scripts at the pins; ttsim smoke (tools/ttsim_smoke.py) on the six P4.9 examples | Host setup |
| P5.4 tt-mlir with LLVM 4efe170d | Done (d0c7533: pin, script, tests) | The install; tests/toolchains/test_tt_mlir_remote.py | Host setup; decisions 2, 4, and 5 |
| P5.5 Polygeist with LLVM 26eb4285 | Finish and audit the stash "P5.5 WIP" (tests, pin, script; unaudited) | The install after P5.4; the remote tests | P5.4's install; g++-12 for the reference source builds |
| P5.6 parse, verify, normalize | lassi/ir code, fakes, fixtures | Remote tests with ttmlir-opt and Polygeist's mlir-opt | P5.4, P5.5 |
| P5.7 cgeist frontend and CPU target | Code and local tests | Fixture capture (naming the workstation), round trips, native runs | P5.5, P5.6 |
| P5.8 TT raiser | LibTooling source and kernels written here | Build against P5.5's clang tree; raise the 13 Tier A kernels | P5.4, P5.5, local tt-metal and ttsim |
| P5.9 program spec | Spec format, refusals, generator | Extraction, ttmetal-host builds, ttsim runs | P5.8; clang-20 at the pinned build |
| P5.10 TT target | Code | EmitC lowering and ttsim runs | P5.4, P5.6, P5.9 |
| P5.12 round-trip driver | `lassi corpus` driver with fakes | One exploratory Tier A dry run | P5.7, P5.10 |
| P5.G gate | none | One run from a clean commit in the WSL clone | Everything above; decision 6 |

Under decision 1 (b), the pins and scripts need no rework beyond the root and stop-line parameters. The stashed Polygeist pin names the same host tools as tt-mlir.pin: clang-17 17.0.6, CMake 3.31.6, ninja 1.10.1, and Python 3.10.12.

### 3.2 Cost (PROJECTED unless marked)

| Build or run | Disk | Time at 8 jobs | Memory | Basis |
| --- | --- | --- | --- | --- |
| tt-metal 5280a9cf and ttsim v1.3.4 | 3.0 GiB plus 172 KiB [MEASURED 2026-09-25, alpha01] | 10 to 30 min plus downloads | Not estimated; keep 8 jobs | results/p4-tt-install: 3 min 44 s in all at 32 jobs, 676 Ninja steps, a 131 s configure [MEASURED, alpha01] |
| tt-mlir stack (LLVM 4efe170d, ttmlir-opt, ttmlir-translate) | 13.25 to 23.30 GiB | LLVM 30 to 90 min, or up to about 2 h if links must run fewer at a time; tt-mlir tools 10 to 45 min | No linker named for LLVM's own build, so GNU ld links its static tools | plans/spikes/p5-mlir-pins.md, Results 5 and :270: upstream CI ran 5,585 Ninja steps in about 19 min at 12 jobs, on an unknown CPU |
| Polygeist stack (LLVM 26eb4285 with clang and mlir) | 11.61 to 19.15 GiB | 1 to 2.5 h | The largest risk: static clang and mlir-opt links with GNU ld | p5-mlir-pins.md Results 5. Also a non-LASSI clang;mlir LLVM build in WSL: 6,967 steps, 9.3 h of summed step time, 2 h 01 min of ninja wall time over three invocations (spike section 3, log) |
| TT raiser (P5.8) | 0.1 to 0.5 GiB | Minutes to tens of minutes | Small | p5-mlir-pins.md |
| Distro, packages, uv, Python 3.10 venv | A few GiB | Under an hour with downloads | n/a | Not estimated in detail |
| P5.G run | Small | 10 to 40 min | Not measured for the repository's Tier A programs or their round trips. The bible's measurement covers the gate's example and the five upstream Tier A examples, run as upstream programs: each peaked at 114 MiB resident or less [MEASURED 2026-09-26, alpha01; docs/BIBLE.md:668, ttsim Facts] | alpha01's exploratory Tier A dry run: 10 trials of 11.9 to 31.0 s, about 201 s, plus 68 s of item tests (PHASE-NOTES P4). Sizing only, never performance |

Totals, all PROJECTED:
- **Disk:** about 28 to 46 GiB of toolchains, against 721.1 GiB free on C: [MEASURED].
- **Time:** about 2 to 6 h of builds run one after another (the sum of the rows).
- **Memory:** the binding constraint. At the defaults WSL has 15.6 GiB plus 4 GiB swap.
  - The old LLVM log's first ninja run averaged 17.3 jobs and stopped after 5,079 of 6,967 steps. The last run averaged 2.0 jobs.
  - That pattern fits an out-of-memory failure, but this is not verified (spike section 3).

The WSL disk is a growing VHDX on C:, and it does not hand space back to Windows by itself (inferred).

### 3.3 What changes in scripts and code (after the decisions; nothing is changed here)

1. **Host profile.** One script sets everything tools/server/gate.py:416-440 (build_env) sets:
   - LASSI_SCRATCH, LASSI_RUNS_ROOT, LASSI_TOOLCHAINS, and LASSI_JOBS (8);
   - TMPDIR, TMP, and TEMP;
   - XDG_CACHE_HOME, UV_CACHE_DIR, PIP_CACHE_DIR, HF_HOME, TIKTOKEN_CACHE_DIR, and CCACHE_DIR;
   - CARGO_HOME, RUSTUP_HOME, DOTNET_CLI_HOME, NUGET_PACKAGES, and npm_config_cache;
   - the scratch root's bin directory at the front of PATH, and PYTHONUNBUFFERED=1 (gate.py:432-439).

   It also sets HOME to the scratch root, since tt-mlir.sh:36-39 checks HOME, and it adds the nice and ionice wrapper (gate.py:445-449). The lassi package needs no change, since it takes its locations from the environment the profile sets:
   - LASSI_SCRATCH, LASSI_RUNS_ROOT, and LASSI_TOOLCHAINS (lassi/core/runner.py:1099, :1122 check_location, :1152);
   - TMPDIR, which must lie under $LASSI_SCRATCH when that is set (runner.py:1914, _checked_tmpdir);
   - HOME, one of the roots every compile hides (runner.py:378, _COMPILE_HIDDEN_VARIABLES);
   - HF_HOME and TIKTOKEN_CACHE_DIR for the model and tokenizer caches (lassi/llm/hf_local.py:167, lassi/scoring/similarity.py:284).
   - /mnt/nvme10 appears in its sources only in comments and docstrings.
   - projects/base.yaml:10 names alpha01's runs root, but $LASSI_RUNS_ROOT takes precedence over it (lassi/core/runner.py:1090-1109).
2. **Install scripts.** The allowed root and the stop line come from explicit host-profile values, in place of the literal /mnt/nvme10/joseph_ufl and 145 GiB.
   - Each value has no default and fails with a clear message when unset (AGENTS.md, rules for [OPEN] choices).
   - This covers tt-mlir.sh, tt-metal.sh, and ttsim.sh, with their tests. cuda.sh and nvhpc.sh change only under decision 8.
3. **Core limit.** tt-metal.sh:34 and ttsim.sh:13 move to `prlimit --pid $$ --core=1:1`, read back as (1, 1), as tt-mlir.sh:57-62 does. PHASE-NOTES (All Phases) already requires this before either script runs again.
4. **Sandbox.** Add -n to sandbox.py:728, :742, :744, and :747, and update SETUP_SCRIPT_SHA256 in tests/executors/test_sandbox.py:2636 (decision 7).
5. **Long-job runner.** It replaces `rx job start --big`. For example, `systemd-run --user` with a MemoryMax around one build at a time, so a runaway link cannot take the desktop down. This is a proposal and is not tested.
6. **Local provenance writer.** It replaces the provenance.json that `rx pull` writes. It records the commit, clean or dirty from the clone's own git, the distro and its version, the WSL kernel, util-linux, the pins, the device, and the date (decision 6).
7. **pyproject.toml:64.** The `remote` marker reads "needs the remote build host through tools/rx.py". It would mean: needs a Linux build host with the sandbox and pinned toolchains (the local distro or alpha01).
8. **tools/status.py.** WORK_ORDER (:37) and the stall rule (:149-155) change under decision 9 (a). If so, also write STATUS.md with LF line endings, which PHASE-NOTES (All Phases) asks for the next time the tool is touched.

### 3.4 What changes in plans and rules for P5

The plan and the notes:
- **plans/p5-ir.md, one planning commit after decisions 1 to 7.** It amends:
  - the constraints, all of which assume alpha01: everything on alpha01's CPU, Agent Rule 17 batches, disk on the scratch root, rx jobs, and the remote practice;
  - the acceptance and Remote lines of P5.4 and P5.5;
  - P5.G's acceptance: "from one clean commit on alpha01", "rx doctor and du are recorded", and "rx pull writes results/p5-gate/ with provenance.json".
  - It also adds the three new tasks in section 3.1.
- **The gate's own words stay as they are.** The Build Roadmap's P5 row names no host.
- **P5.4 stays OWNER** until the new tasks are DONE. Its OWNER item changes from OQ-054 to decision 2's install item. `tools/status.py set` takes `--depends` (tools/status.py:284-289), so the new tasks can become P5.4's dependencies.
- **PHASE-NOTES, All Phases.** The lines that route compilers and simulators through rx, rx jobs, rx doctor, and the scratch cap are rewritten once decision 1 sets the local route. The excess-files line follows decision 11.

Rules:
- The bible's Toolchain Pins and Sandbox text and AGENTS.md's Results and Phase Gate sections change as listed in section 6. Agent Rules 7 and 17 change only if decision 3 or 4 chooses a rule change.

### 3.5 Gate feasibility

The gate reads: "Every Tier A reference pair round-trips on ttsim; per-kernel gate rates reported (Tier B moved to P18 by the owner, 2026-10-11)". It names no host. It is feasible in WSL in principle, once sections 3.1 to 3.4 are done (inferred: nothing was installed or run):

- ttsim needs only the C library, at GLIBC_2.14 at most (bible, ttsim Facts). WSL has 2.39.
- The ttsim executor sets TT_METAL_THREADCOUNT to the run's CPU count (lassi/executors/ttsim.py:278), and a run's CPU count is 16 (RUN_CPUS, lassi/core/stages.py:396). That equals WSL's 16 threads.
- The bible measured the gate's example and the five upstream Tier A examples, run as upstream programs, at 114 MiB resident or less each on alpha01 (docs/BIBLE.md:668, ttsim Facts). The repository's Tier A host programs and their round trips were not measured.
- The C++ side runs natively. That needs the gcc pin for the reference source builds, which is met under decision 1 (b).

Still unmeasured, or a risk:
- **Memory** for the LLVM-scale links (decision 5).
- **The CPM package list.** tt-metal's configure fetches 25 CPM packages, and their sources must equal toolchains/tt-metal-cpm-sources.txt line for line. An upstream tag that moved since 2026-09-25 would be refused, by design, and would need a pin decision. Not checked.
- **Superseded packages.** apt.llvm.org and jammy-updates may later replace the pinned builds. A proposal: keep the fetched .deb files and their sha256 under the scratch root.
- **cgeist's reach** on the Tier A C++ references, which is unchanged by the host. A reference it cannot raise still goes to the owner queue, as P5.7 plans.
- **Crash dumps.** Whether the 1-byte core limit stops WSL's crash handler (decision 3).

## 4. Other phases and the proposed local work order

### 4.1 Phases with a local route

| Phase | Gate names a device? | Local now (Windows, no decision) | Local later (WSL) | Waits on |
| --- | --- | --- | --- | --- |
| P11 Dataflow Dialect | No: "Tier A pairs round-trip through df", on ttsim | Design notes as owner-queue proposals (bible question 7); a spike reading the source for question 13 (does tblgen-to-irdl cover every df construct?) | ODS on P5.4's LLVM; the gate after P5.8 to P5.12 | Decision 9; P5 |
| P12 Language Frontends | No: "xlang-v0 programs raise to the hub and round-trip natively in every language" | Planning; a frontend-pins spike (documentation only); the Python-subset frontend with unit tests | Rust MIR, C# Roslyn, Fortran, LLVM import. None is installed in WSL: no rustc, cargo, dotnet, flang, gfortran, or clang is on PATH or known to dpkg [MEASURED 2026-10-15, workstation; spike, "Probes of the planning round (2026-10-15)"]. Native runs in the sandbox; the gate after P5.6 and P5.7 | Decisions 1, 2, and 9; P5.6 and P5.7 |
| P16 Adversarial | No, except its training method (P8) | Adversary agent, fuzzer, input-contract checks. lassi/agents does not exist yet (the bible's layout lists it, docs/BIBLE.md:174), and the runner refuses adversary sections (lassi/core/runner.py:365) | Seeded faulty C++ candidates run natively in the sandbox, judged by binary_io | Decision 12; the sandbox; the gcc pin |
| P17 Portable Stack (P17.12, P17.13, P17.G) | Part (b) needs a GPU host | None | P17.12 as a spike on the RTX 3080 (see below) | Decision 8; the cuda extra and cuda@12.6.3 under a local root |

Note (2026-10-18): the owner named the df dialect and TurboQuant separate projects, to be incorporated later (OQ-066; plans/LESSONS.md, Owner directions). The P11 row above no longer applies: P11 is outside the work order and BLOCKED, with its plan frozen on branch p11-df. The rest of this section stands as written, except that P11's design notes no longer run beside P5 (section 4.2, step 5) on 2026-10-15.

The P17.12 spike would measure what CUDA needs in the sandbox under WSL: /dev/dxg and /usr/lib/wsl/lib. The current code fits a Linux host, not WSL:
- the cuda probe requires dev/nvidiactl (lassi/executors/devices.py:183);
- the gpu executor's table binds /dev/nvidia* (lassi/executors/gpu.py:102-104);
- the nvml profiler finds GPUs under /proc/driver/nvidia/gpus (lassi/profilers/nvml.py:12).

None of those exists in WSL: no /dev/nvidia* node, no /dev/nvidiactl, and no /proc/driver/nvidia [MEASURED 2026-10-15, workstation; spike, "Probes of the planning round (2026-10-15)"]. So P17.13 would be a [DESIGN] change.

P17's part (a) is done (P17.11, PR 5). Its gate's rerun clause, "on alpha01's CPU", needs restating only under decision 8 (c).

### 4.2 Proposed local work order

1. **Now, with no decision needed** (Windows, then push). Steps (a) and (b) are task work: each needs a STATUS task ID, added with `uv run tools/status.py add`, before it starts, since AGENTS.md commits once per task under a `P<N>.<k>:` subject.
   - (a) **Pin the CI runners before 2026-10-19.** Both workflows run on ubuntu-latest (.github/workflows/cpu-tests.yml:16, text-policy.yml:13). A runner notice says ubuntu-latest moves to Ubuntu 26 from that date (PHASE-NOTES P17; not verified). Pin ubuntu-24.04 or move to newer action majors, then rerun.
   - (b) **Fix the stale P17 text.** The P17 phase Note says P17.11 is OWNER and no PR is open, while task P17.11 is DONE with PR 5. The bible's work-order sentence still says P17.11's batch waits for approval.
   - (c) **Record the direction in the bible**, through the bible-sync procedure:
     - the spike's drafted Workstation row and NVIDIA row addendum;
     - the util-linux 2.39 Sandbox bullet;
     - the RNGD and Tenstorrent rows;
     - a Decision Log entry; the count becomes one hundred and forty-one.
     The docs connector must be up (owner, 2026-10-12).
   - (d) **Commit the spike and this plan**, and file section 7's decisions in the owner queue. Then set the P5 tasks that wait on decision 1 OWNER, as AGENTS.md requires for a choice (decision 9).
2. **Owner decisions** 1 to 7, then 8 to 14.
3. **Host setup and qualification.** The new tasks in section 3.1:
   - installs per decision 2, and the sandbox change;
   - the profile and script parameters, and the WSL clone;
   - the remote-marked tests run there as the host's acceptance, every result naming the workstation. The hipcc and NVIDIA ones need toolchains this plan does not install.
4. **P5 locally**, the critical path: the tt-metal and ttsim rebuild, P5.4, P5.5, then P5.6 to P5.12 and P5.G. One build runs at a time in the background, with each task's Windows half done between builds.
5. **Beside P5's builds** (decision 9): P12's planning, its pins spike, and the Python-subset frontend, once P5 is stalled; P11's design notes and the question 13 spike too under decision 9 (a), else once P12 is DONE, GATE-OWNER, or stalled.
6. **P12's toolchain frontends** once their installs are approved, and P12's gate after P5.7.
7. **P16 part (a)**, if decision 12 opens it.
8. **P17.12** on the RTX 3080, then P17.13 and a restated P17.G, if decision 8 allows.
9. **Everything else stays BLOCKED** (section 5).

Until decisions 1 to 5 are answered: no install, no .wslconfig change, and no LLVM-scale build. Until decision 6 is answered, no local result is [MEASURED].

## 5. What stays blocked and why

| Phase | Blocker as written | Why there is no local route | Revisit when |
| --- | --- | --- | --- |
| P3 RNGD Serving | Smoke per arm on RNGD | No RNGD (the direction) | RNGD returns |
| P6 DF Zero-Shot | MLIR and source arms on RNGD | RNGD. Also, tt-pairs-v0 has no eval item until P5.18 builds Tier D (OQ-066) | RNGD returns |
| P7 Offline Training | RNGD sampling, fxb export, training GPUs | RNGD and GPUs. No train-split item exists: Tier A is unassigned (OQ-025) | Hardware, then split decisions |
| P8 Online Training | MI300X (vLLM rollouts) | No MI300X (OQ-040) | MI300X access |
| P9 LASSI-EE | MI300X power protocol | As P8. A CUDA-arm port on the RTX 3080 would be a restated gate | After P17.13, if decision 8 leads there |
| P10 Full LASSI | A100 host; arms A1 to A4 on RNGD | No A100 and no RNGD. Arm B0, a 33B model at 8 bits (PROJECTED at about 35 GB), does not fit in 10 GiB | An A100 host |
| P13 Cerebras Target | Cerebras SDK access (OQ-004) | No SDK. Its simulator ships as an Apptainer container, and WSL has no working Docker (spike section 3) and no Apptainer or Singularity (spike, "Probes of the planning round (2026-10-15)") | Cerebras access arrives (OQ-004, CLOSED; decision 14 confirms it) |
| P14 Furiosa Target | Reference kernels on RNGD | No RNGD | RNGD returns |
| P15 Judges | Calibration on MI300X measurements (P9) | P9 | After P9 or a restatement |
| P18 TurboQuant Tier B, with P5.13 and P5.14 | The owner's word on TurboQuant Hotswap (OQ-048) | Unchanged by the direction. Once opened, it needs P5's local route | The owner's word |

Several of these phases have fragments that need no hardware:
- P3: the `furiosa` alias;
- P7: the training table and hooks;
- P9: the EE stages with the mock model;
- P13: xDSL csl emission;
- P14: a documentation spike on question 11;
- P15: the judge classes.

This plan schedules none of them. None reaches a gate, and each phase stays BLOCKED until the owner records otherwise in its Note (AGENTS.md, Work Order).

## 6. Required changes (listed, not made)

### 6.1 Bible (bible-sync procedure; the mirror is at master revision 276)

These are factual edits that need no decision (work order step 1):
- **Environment State:** a Workstation row, the NVIDIA GPU row's addendum, and the RNGD and Tenstorrent rows stating the 2026-10-15 direction. The drafts are in plans/spikes/local-host.md, "Proposed bible edit".
- **Sandbox:** the util-linux 2.39 bullet (drafted in the spike).
- **Build Roadmap:** the work-order sentence, which is stale on P17.11, and the P3 and P14 Blocked-by cells ("the cards are on alpha01").
- **Risks And Questions:** the rows that call RNGD reachable.
- **Decision Log:** one entry dated 2026-10-15; the count sentence becomes "One hundred and forty-one".

These follow the decisions:
- **Agent Rules 7 and 17:** only if decision 3 or 4 picks a rule change. Otherwise Host Facts gains a Workstation part covering the scratch root, cap, stop line, VHDX drive, crash dumps, and build approvals.
- **Agent workflow bullet:** "They reach alpha01 only through tools/rx.py".
- **Toolchain Pins:**
  - the install-location and space-budget sentences that name alpha01;
  - the gcc, ttmetal-host, and MLIR-stacks bullets;
  - each install's record once it runs.
- **Sandbox:**
  - "Sandboxed runs start with systemd-run --user through the gate";
  - "Mechanism on alpha01";
  - the per-trial build directory under /mnt/nvme10;
  - the gpu program runs bullet's "A spec without an exposure runs the setup script byte for byte as before task P17.5", which becomes untrue once -n lands (decision 7).
- **Execution Backends:** the ttsim and native rows, which cite alpha01's installs, and the gpu NVIDIA row under decision 8.
- **Build Roadmap:** P17's gate part (a), "on alpha01's CPU", under decision 8 (c); the work order under decision 9 (a); P16 under decision 12.
- **Environment State and Host Facts headers:** their alpha01-only framing.

### 6.2 AGENTS.md

- **Local Execution** (decisions 1 to 5): a new section beside Remote Execution covering the distro, roots, profile, long-job runner, disk floor, and approvals.
- **Results and Phase Gate** (decision 6): the local provenance procedure.
- **Work Order:**
  - parallel phases (decision 9 (a));
  - P16 (decision 12);
  - "until a GPU host exists" (decision 8).
- **Remote Execution:** the excess-files inventory sentence, under decision 11.
- **Phase Planning, step 2:** "remote needs" becomes "host needs".
- **Run Reports:** "remote jobs still running" also covers local jobs.
- **Agent Rules 7 and 17:** restated only if the bible changes them.

### 6.3 STATUS.md (only through tools/status.py), plans, and notes

STATUS.md:
- **P5.5 to P5.10, P5.12, and P5.G:** OWNER on decision 1's item once it is filed (AGENTS.md, Owner Queue; decision 9), and back to READY when it is answered.
- **P5.4:** OWNER on decision 2's install item instead of OQ-054, with the new host tasks added as its dependencies (`set --depends`).
- **New P5 tasks:**
  - local host setup and the evidence procedure;
  - the sandbox on util-linux 2.39;
  - tt-metal and ttsim rebuilt locally.
- **P5.5's Note:** the job waits for P5.4's local install, not OQ-054.
- **P17 phase Note:** fix the stale text (`tools/status.py phase P17 ACTIVE --note ...`).
- **P16 phase Note:** under decision 12.

Plans and notes:
- **plans/p5-ir.md and PHASE-NOTES, All Phases:** as in section 3.4.
- **pyproject.toml:64:** the marker text (section 3.3).
- **plans/LESSONS.md, Status and planning:** its lesson says tools/status.py cannot change a task's title or dependencies. `set` now takes `--title` and `--depends` (tools/status.py:284-289), so that lesson is stale.

### 6.4 Owner queue

- File decisions 1 to 14 as items. Numbers are assigned at filing (plans/LESSONS.md).
- OQ-054 and OQ-045 follow decision 10.
- This plan changes nothing about OQ-044 and OQ-051, which are owner actions.

## 7. Owner decisions

Note (2026-10-18): the owner answered these decisions on 2026-10-15 (OQ-055), and on 2026-10-18 named the df dialect and TurboQuant separate projects, to be incorporated later (OQ-066). Decision 9 (a)'s P11 half is superseded, and its stall rule stands; P11 (section 4.1) and P18 (section 5) stay BLOCKED outside the work order. The P11 text below (decision 1 (a), decision 9, decision 11, and the P11 proposals left to phase planning) is kept as the record of 2026-10-15. Of the choices left to P12's planning, xDSL as a pinned dependency is deferred to P13's planning (OQ-059, OQ-066).

**1. Where LASSI builds and runs locally** (spike draft A; this replaces OQ-054's route)
- (a) Wait for alpha01.
  - Nothing is installed. P5 does only Windows halves after P5.11.
  - P11's ODS, P12's round trip, and P16's runs wait too.
- (b) A dedicated Ubuntu 22.04 WSL distro that mirrors alpha01, which runs Ubuntu 22.04.5 (results/p4-tt-install).
  - Every pinned host-tool string is served today (section 2.3), so no pin changes. P4's alpha01 evidence stays like for like.
  - Its util-linux is 2.37.2, as on alpha01, so the committed sandbox setup probably runs unchanged (inferred; not run).
  - It adds a second distro on the owner's machine. Systemd must be enabled in it, and the sandbox re-qualified there.
- (c) Re-pin to the existing Ubuntu-24.04 distro.
  - Pins change for gcc, tt-metal, tt-mlir, Polygeist, and ttmetal-host (Agent Rule 10, with Decision Log entries).
  - Noble's clang-20 is a different build, so P4's evidence is no longer like for like.
  - Decision 7 (a) is needed first. The distro is shared with other projects (~/mlir_dev, ~/xdsl).
- Recommendation: (b).

**2. Installs on the workstation: who does them, and with what approval**

The first batch, under 1 (b):
- the distro, by `wsl --install`, or by `wsl --import`, which lets the owner choose the drive for its disk (doc; inferred);
- from apt: clang-17, lld-17, clang-20, and lld-20 from apt.llvm.org jammy; g++-12; ninja-build; python3.10 with venv and dev; git, curl, make, and pkg-config; libhwloc-dev, libnuma-dev, libtbb-dev, libcapstone-dev, libssl-dev, and xz-utils;
- CMake 3.31.6 from Kitware's release archive, checked by sha256, into /usr/local;
- uv and Python 3.10 for the control plane.

Later batches: cuda@12.6.3 under a local root (decision 8), and rustup nightly, a Linux .NET SDK, and flang (P12).

- (a) The owner installs each batch by hand from the list. No agent touches system state.
- (b) An agent writes one install script per batch, with exact versions, sha256 hashes, sizes, and the commands. The owner reviews it and runs it; sudo needs the owner's password. The script and its output become the install record.
- (c) An agent runs each batch itself after the owner's yes. Unattended, that needs a sudo rule with no password, which is itself a system setting.
- Recommendation: (b). The owner keeps control of system changes, the provenance gets an exact record, and no sudo rule is needed.

**3. Host policy for the workstation** (the counterpart of Agent Rule 7)
- (a) Amend Agent Rule 7 to name the workstation. That is an Agent Rule change, mirrored in the bible.
- (b) Write the policy in Host Facts and an AGENTS.md Local Execution section, and leave the rules unchanged. Proposed content:
  - the scratch root inside the LASSI distro's ext4 disk, never on /mnt/c or another Windows drive mount;
  - that disk on C:, which is NVMe with 721.1 GiB free, or on Z:, a SATA SSD with 578.3 GiB free;
  - a cap of 150G and a stop line of 145G for the scratch root, alpha01's numbers, so the scripts keep one rule;
  - a floor of 100 GiB free on the drive that holds the disk (a proposal; the owner picks the number);
  - nothing in the distro's /tmp, /var/tmp, /opt, or a HOME outside the scratch root;
  - a dump in %TEMP%\wsl-crashes counts as a write outside the scratch root. Host qualification checks the 1-byte core limit with one deliberate crash of a tiny program, which leaves at most a small dump if the limit fails (inferred).
- (c) No policy. Builds are bounded only by the scripts' own checks, and nothing stops one from filling C:.
- Recommendation: (b).

**4. Approval for heavy local work** (the counterpart of Agent Rule 17)
- (a) Every build or long run waits for the owner's yes per batch, as on alpha01. Unattended sessions stop at each of P5's builds.
- (b) The installs (decision 2) and the first LLVM-scale build wait for the owner's yes. After that, builds and runs at pinned commits may run unattended, under these limits:
  - one at a time, niced;
  - inside a `systemd-run --user` scope with a memory cap;
  - each one reported in the run report.

  The desktop is busy for hours at a time (PROJECTED 2 to 6 h in all).
- (c) Everything runs unattended once the installs are approved.
- Recommendation: (b). The first build measures memory and time on this machine before any build runs unattended.

**5. WSL memory for LLVM-scale links**
- (a) Keep the defaults (15.6 GiB plus 4 GiB swap) and build at 8 jobs or fewer. Static links with GNU ld may run out of memory and cost a rerun at fewer jobs. No pin changes.
- (b) The owner raises the memory in .wslconfig to about 24 GiB of the 31.9 GiB for the build window. The figure is a proposal; the owner picks it. That is a Windows setting, and WSL restarts. Windows keeps about 8 GiB while a build runs. No pin changes.
- (c) Add LLVM_PARALLEL_LINK_JOBS to the LLVM flags. That changes tt-mlir.pin's committed flags (Agent Rule 10, with a Decision Log entry).
- Recommendation: (b), or (a) if the owner wants the desktop's memory left alone.

**6. Evidence for local runs** (Agent Rule 1; AGENTS.md Results and Phase Gate; P5.G's acceptance)
- (a) Local runs are exploratory only, and [MEASURED] waits for alpha01. No gate can pass locally, and P5 cannot finish.
- (b) [MEASURED] comes from a clean commit in a WSL clone, cloned from /mnt/c/dev/lassi with no network.
  - The provenance manifest names the workstation, the distro, the WSL kernel, util-linux, the pins, the device (the host CPU, or ttsim v1.3.4 on tt-metal 5280a9cf), and the date.
  - Results are copied into results/<name>/ in the Windows tree and committed there, and are never presented as alpha01's.
  - P5's gate evidence from the workstation stands beside P4's alpha01 evidence for the same pins.
- Recommendation: (b).

**7. The sandbox change for util-linux 2.39**
- (a) An agent task now:
  - -n on the four `mount --move` lines, the setup hash pin updated, the bible's Sandbox text corrected, and a Decision Log entry;
  - the remote sandbox tests run in WSL from a clean commit, labeled with the workstation.

  It works under every option of decision 1; it was measured on 2.39.3 only, and that run was exploratory. alpha01's util-linux 2.37.2 with -n stays unmeasured until alpha01 returns, when one sandbox run re-checks it.
- (b) No change; rely on the 22.04 distro's util-linux 2.37.2. No unmeasured change reaches alpha01. The sandbox keeps failing on any util-linux 2.39 host, which rules out decision 1 (c).
- Recommendation: (a). -n skips only libmount's userspace mount-table update, which the setup never reads (inferred from mount(8) and libmount's documentation; spike section 4).

**8. The RTX 3080 as P17's GPU host** (spike draft B)
- (a) No. P17.12, P17.13, and P17.G stay BLOCKED (OQ-040).
- (b) P17.12's spike only. It measures what CUDA needs in the sandbox under WSL, with no model inference or training, and then the owner decides again. It uses the owner's display GPU briefly, and needs the cuda extra and cuda@12.6.3 under a local root (decision 2).
- (c) All of part (b): the gpu executor, the nvml profiler, hf_local on cuda, and short training runs, with P17.G restated (part (a) rerun on the workstation CPU, part (b) on the RTX 3080). It needs a [DESIGN] change to the cuda probe, the gpu table, and the nvml lookup (section 4.1). Models must fit in 10 GiB, and the card is neither an A100 (P10) nor an MI300X (P8, P9).
- Recommendation: (b).

**9. Work order under local-only work**

Today `tools/status.py next` returns none. P5's READY tasks all depend on P5.4, which is OWNER, so P5 is not stalled (tools/status.py:149-155) and P12 cannot advance (:37).

AGENTS.md already gives a route, under every option below. For a choice, it says to "set dependent tasks OWNER, and continue with unblocked work" (Owner Queue, rules for [OPEN] items). Once decision 1 is filed:
- the P5 tasks that wait on it (P5.5 to P5.10, P5.12, and P5.G) are set OWNER;
- P5 then has DONE tasks and none READY or ACTIVE, so it is stalled (tools/status.py:149-155), and `next` advances P12 (:37), with no tool change;
- P11 still follows P12 in WORK_ORDER, and P5's Windows halves wait with their OWNER tasks;
- the tasks go back to READY with `set` once decision 1 is answered.

The options:
- (a) Beyond that route, let P11's design notes run beside P5 and P12, as the bible's work-order sentence already says ("with P12 frontends and P11 design notes in parallel"). That means a change, with tests, to WORK_ORDER and the stall rule, and to AGENTS.md Work Order.
- (b) The route alone: P5's waiting tasks become OWNER, not BLOCKED, on decision 1's item, so P5 counts as stalled and P12 advances. No tool change. P11 waits until P12 is DONE, GATE-OWNER, or stalled.
- (c) Leave the tasks READY until decision 1 is answered. That departs from the AGENTS.md route, and unattended sessions stop until then.
- Recommendation: (a). The route alone, (b), already lets P12 advance with no tool change; (a) adds P11's design notes beside it, as the bible's work-order sentence says.

**10. OQ-054 and OQ-045, which the direction makes moot**
- OQ-054 (P5.4's install on alpha01, answered "Not yet"):
  - (a) Close it as superseded, and move P5.4 to decision 2's install item.
  - (b) Keep it open for alpha01's return.
- OQ-045 (the real-model Tier A run on RNGD):
  - (a) Hold it OPEN until RNGD returns.
  - (b) Redefine it as a local-model run (hf_local on the CPU, or the RTX 3080 under decision 8) with local ttsim. That is a new run definition, with its own item.
- Recommendation: close OQ-054, since its batches name alpha01 (a new item can be filed if alpha01 returns), and hold OQ-045.

**11. The standing alpha01 excess-files inventory** (owner, 2026-10-11; PHASE-NOTES, All Phases; AGENTS.md, Remote Execution)

It obliges the first phase planned after P5 (P12 or P11) to run a probe on alpha01, which the direction forbids.
- (a) Defer it until alpha01 access returns.
- (b) Drop it.
- (c) Keep it, so the next plan carries a task that stays BLOCKED.
- Recommendation: (a).

**12. P16's split**
- (a) Keep P16 BLOCKED.
- (b) Record in its Note that the P4 blocker is cleared, and split the phase:
  - part (a), local: the adversary agent, the fuzzer, input-contract checks, and the gate as written;
  - part (b): the adversarial training method, BLOCKED on P8.

  P16 then joins the work order after P12. A sub-question: the bible's Adversarial Setups say "Confirmed counterexamples join the held-out input set" (docs/BIBLE.md:826), while Agent Rule 5 forbids any training, prompt tuning, or corpus harvest on eval items. May confirmed counterexamples join the held-out inputs of unassigned or eval items, or only synthetic fixtures outside every suite?
- Recommendation: (b), with synthetic fixtures only.

**13. Gates that name hardware**
- (a) Restate none now, so P3, P6 to P10, P14, and P15 stay BLOCKED. Revisit P9 and P15 as RTX 3080 ports only after P17.13 shows sandboxed GPU runs with power readings under WSL.
- (b) Restate some now, for example P9 as a CUDA port on the RTX 3080. That sets new gates before any measured local GPU path exists.
- Recommendation: (a). P17.G is restated only under decision 8 (c).

**14. P13 Cerebras: a confirmation, not a new question**

OQ-004 (Cerebras SDK Access) is CLOSED with the owner's answer: "Cerebras access is incoming. Don't worry about it right now." It was applied on 2026-09-23: P13 stays BLOCKED until access arrives (plans/OWNER-QUEUE.md). The direction of 2026-10-15 changes nothing there.
- (a) Confirm OQ-004's answer: P13 stays BLOCKED until the SDK arrives, and this plan schedules none of its SDK-free fragments.
- (b) Open an SDK-free part anyway: xDSL csl emission and a csl-pairs-v0 manifest from the public csl-examples. It needs an xDSL pin, and it goes beyond "Don't worry about it right now".
- Recommendation: (a). P13's gate needs the SDK's simulator container.

Left to phase planning, not asked now. These go to the owner queue when those phases are planned:
- P12's scope choices:
  - which phase builds cgeist's CUDA path (PHASE-NOTES P12);
  - the xlang-v0 items and their splits (bible question 10);
  - xDSL as a pinned dependency.
- P11's design proposals (bible questions 7 and 9).
