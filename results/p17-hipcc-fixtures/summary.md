# P17.6: hipcc-gfx942 remote test and stderr fixtures

[MEASURED] provenance.json (rx run 20261006-121638-desktop-8r113ei-detached-f03fa62a-961a):

- commit: f03fa62 (f03fa62a5207436bbd8131911576a9382a0e0d42), clean tree: `dirty` false and `snapshot_of` null in provenance.json and in the capture manifest, and the slot's `git status --short | wc -l` printed 0 (the run's first output line);
- host: alpha01, Ubuntu 22.04.5 LTS, Python 3.10.12, 256 CPUs; start 2026-10-06T12:16:38-07:00, end 2026-10-06T12:16:55-07:00; rc 0, and the command's last line was `remote_rc=0 capture_rc=0`;
- toolchain pin: toolchains/hipcc.pin, VERSION 7.12.60610-2bd1678d3d, EXECUTABLE /opt/rocm/core-7.12/bin/hipcc, FLAGS `--offload-arch=gfx942 -Wall -O3`, EXPECT_VERSION `HIP version: 7.12.60610-2bd1678d3d` (provenance.json pins and the manifest's pin_files hold the same values);
- compiler: /opt/rocm/core-7.12/bin/hipcc; its version check, `/opt/rocm/core-7.12/bin/hipcc --offload-arch=gfx942 --version`, run through the compile sandbox before the first build, printed `HIP version: 7.12.60610-2bd1678d3d`, `AMD clang version 22.0.0git (https://github.com/ROCm/llvm-project.git c849bc16b0e49951d313756f20b73c2b28d321d7+PATCHED:9a6ac45c97a1e511db838c5b46257324d2de1780)`, `Target: x86_64-unknown-linux-gnu`, `Thread model: posix`, `InstalledDir: /opt/rocm/core-7.12/lib/llvm/bin` and exited 0; the pin's EXPECT_VERSION is the first of those lines;
- compile environment: LANG, LC_ALL, PATH (LANG=C, LC_ALL=C); no HOME, no loader variable, and no HIP or ROCm variable; no ROCm environment script was sourced;
- scratch root free: 394.8 GB before and 394.8 GB after; root filesystem free 241.3 GB (provenance.json);
- device: alpha01's CPU only. No GPU device was opened: every build named its target, `--offload-arch=gfx942`, the two traced commands started no device-query program and touched no GPU path (remote tests below), and no built program ran.

Command (provenance.json, cmd):

```
uv sync --quiet && git status --short | wc -l && { LASSI_REQUIRE_SANDBOX=1 uv run pytest -q -x -p no:cacheprovider -m remote tests/toolchains/test_hipcc_remote.py; t=$?; c=skipped; if [ "$t" -eq 0 ]; then uv run python tools/capture_toolchain_fixtures.py --fixtures tests/toolchains/fixtures/hipcc; c=$?; fi; echo "remote_rc=$t capture_rc=$c"; [ "$t" -eq 0 ] && [ "$c" = 0 ]; }
```

## Precondition, read-only

Before the run, a read-only grep of the hipcc binary, which ran nothing from the ROCm tree, `grep -c rocm_agent_enumerator /opt/rocm/core-7.12/bin/hipcc`, run with other read-only checks (an `ls` of target.lst beside hipcc and a listing of the KFD topology nodes; plans/OWNER-QUEUE.md, OQ-046's evidence) in rx exec 20261006-115446-exec-bd6f, counted 1: the binary names the enumerator that hipcc's source runs when no target is given, so the explicit target in the version check and in every build matters on this build.

## Remote tests

The remote tests, run with LASSI_REQUIRE_SANDBOX=1 so that none could skip, and with -x, printed `6 passed in 5.48s`, both trace tests included:

- the version check, traced under strace -f in the compile sandbox: exactly one command, `hipcc --offload-arch=gfx942 --version`, which printed EXPECT_VERSION and started the pinned hipcc and a clang, no device-query program (amdgpu-arch, offload-arch, rocm_agent_enumerator, rocminfo, rocm-smi, amd-smi), and touched no GPU path;
- the check ran in the compile sandbox with exactly PATH, LANG, and LC_ALL, and a pin copy with another EXPECT_VERSION was refused;
- the kept program (tests/toolchains/fixtures/hipcc/sources/hip_clean/main.hip), compiled and linked under strace -f through the toolchain's own compile runner: exit status 0, no device-query program, and no GPU path. This is the first measured link of a hipcc program in the compile sandbox, and it needed no variable beyond the sandbox's;
- the same program built through build(): an artifact `main` and no error;
- the artifact's `.hip_fatbin` section holds a clang offload bundle with one host entry and exactly one device entry, whose id ends in `--gfx942`: an ELF code object with e_machine EM_AMDGPU and the gfx942 machine flag; the artifact's dynamic section lists libamdhip64.so as NEEDED.

Above, "no GPU path" is what the trace check reads: no absolute path strace printed starts with one of the test's GPU_PATH_PREFIXES (/dev/kfd, /dev/dri, /sys/class/kfd, /sys/class/drm, /sys/devices/virtual/kfd, /sys/devices/pci, /sys/bus/pci, /sys/module/amdgpu, /proc/bus/pci, /proc/driver). A path given relative to a directory descriptor is not resolved (tests/toolchains/test_hipcc_remote.py, traced_calls), so the check would miss one opened that way.

## Capture

The capture (the manifest dated 2026-10-06T12:16:45-07:00) built each case once with the hipcc-gfx942 toolchain, as the stage runner builds it, in a fresh workdir. The diagnostic counts are the manifest's, which the parser of commit f03fa62 gave:

| Case | Exit status | Stderr bytes | Diagnostics parsed | Stderr sha256 |
| --- | --- | --- | --- | --- |
| hip_clean | 0 | 0 | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| hip_device_undefined | 1 | 693 | 1 | `167e2b92586a815fe1109f068269d224a2dbbd2facf6d0fe49e15670ab513df2` |
| hip_host_calls_device | 1 | 527 | 2 | `64b9669d93adcbf4812d247ed2acb404609dc5db74a31da2c726e32f660cfc85` |
| hip_kernel_calls_host | 1 | 491 | 2 | `d11b507c4ef477992133262708234c38591ae2771ab04d44efb611876fbdb8b1` |
| hip_linker_error | 1 | 480 | 2 | `78caa2810c40ffaf0332a7fe89840dcb3c44339eddab08aba15c0a4c76ddc5af` |
| hip_missing_header | 1 | 342 | 1 | `635526b3fd59719fbb03977a50af640bc0919499259d169fdd8388be88abfa0b` |
| hip_undeclared_identifier | 1 | 349 | 1 | `405f8bcb460d09a0739303fabf3fba4c099cd6194c61e58ca3d2f449130decb2` |
| hip_unused_variable | 0 | 330 | 2 | `b4c4a8de28c5c2f92506a72fd64cb909a2d84883576bae15713bac1f5a828ee8` |

- tests/toolchains/fixtures/hipcc/<case>.stderr and captures.json are byte-identical to the capture's `<case>.stderr` files and manifest.json (compared with cmp after the pull), and each stderr's sha256 is the manifest's. All eight are plain ASCII with LF line endings; hip_clean.stderr is empty.
- The sources the capture built (work/<case>/main.hip in the pull) are byte-identical to tests/toolchains/fixtures/hipcc/sources/, and each work/<case>/compile.stderr is byte-identical to the case's `.stderr`.
- Each <case>.json was derived by hand from its stderr, and the parser gives exactly those lists (tests/toolchains/test_hipcc.py).
- The one parser change: the gfx942 device link fails in lld, which names itself `lld:` (`lld: error: undefined hidden symbol: device_helper(int)`), not `ld.lld:` as the host link does. The parser of commit f03fa62 read no such line, so the manifest counts 1 diagnostic for hip_device_undefined; lassi.toolchains.hipcc now reads it with its own DEVICE_LLD pattern, and the case parses into 2. ttmetal-host's patterns are unchanged.
- What else the capture showed: a warning in a kernel is printed by the gfx942 pass and again by the host pass, and both are kept; a compile error stops the build after the gfx942 pass, so it is printed once, with `1 error generated when compiling for gfx942.` and no host summary; the host and device target errors are `no matching function for call to '<name>'` errors at the call, each with a note at the declaration; the host link's linker is ld.lld, with the driver line `clang++: error: linker command failed ...`, and the device link's driver line is `clang++: error: amdgcn-link command failed ...`; every failed build ends with hipcc's own line, `failed to execute:` and the clang++ command it ran, which parses to nothing; no line comes from clang-offload-bundler. tests/toolchains/fixtures/hipcc/README.md gives the details.
- hip_linker_error.stderr and hip_device_undefined.stderr name the capture's workdir and clang's temporary objects in it, `@lassi-tmp/main-8098dc.o` and `@lassi-tmp/main-gfx942-72a9c2.o`, in lld's `>>> ` lines; that text changes on a recapture. The other six name no per-run text.

## The kept program's dynamic section, read-only

After the run, a read-only exec read the clean build's dynamic section without running it, `readelf -d -W /mnt/nvme10/joseph_ufl/lassi-runs/fixture-captures/20261006-121638-desktop-8r113ei-detached-f03fa62a-961a/work/hip_clean/main | grep -E "NEEDED|RUNPATH|RPATH"` (rx exec 20261006-121713-exec-a66b). It printed five NEEDED lines and no RUNPATH or RPATH line:

| Tag | Shared library |
| --- | --- |
| NEEDED | libamdhip64.so.7 |
| NEEDED | libstdc++.so.6 |
| NEEDED | libm.so.6 |
| NEEDED | libgcc_s.so.1 |
| NEEDED | libc.so.6 |

So a hipcc-built program names libamdhip64.so.7 without a path to find it, and the sandbox passes no LD_LIBRARY_PATH. Nothing here runs one; how a run finds the library is P17.13's decision (plans/PHASE-NOTES.md, P17 section).
