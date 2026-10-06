# P17.1: the CPU framework set and hipcc for gfx942 on alpha01

[MEASURED] provenance.json in this directory, the rx run record (rx 20261005-204140-desktop-8r113ei-detached-e805c6d4-54d2). Every value below comes from it or from the report files it names, unless the line names another source. plans/spikes/p17-frameworks.md gives the design, the reading of these files, and the parts of P17.1 that were read from metadata and source rather than run.

## Run

- rx id 20261005-204140-desktop-8r113ei-detached-e805c6d4-54d2, state done, rc 0; start 2026-10-05T20:41:40-07:00, end 20:42:07-07:00 (27 s); timeout 7200 s (provenance.json).
- Commit e805c6d (e805c6d4af0a16e6458c4971362659b836112c50), run in the detached clean worktree slot desktop-8r113ei-detached-e805c6d4 (provenance.json: branch detached-e805c6d4, slot): a clean tree, read from the rx record (provenance.json: dirty false, snapshot_of null); batch.txt also prints "tree clean".
- Command: `bash plans/spikes/p17-frameworks/run.sh` (provenance.json, cmd).
- Host: alpha01, Ubuntu 22.04.5 LTS, kernel 6.6.29+main+3.0.0r1-amd64-gio-epilmore-dev+, nproc 256; system Python 3.10.12; uv 0.12.17 (batch.txt).
- Device: the host CPU (cpu_capability AVX512, imports.json). No GPU was opened: the imports and every hipcc step ran in the P0.16 sandbox, whose private /dev binds only null, zero, full, random, urandom, and tty (lassi/executors/sandbox.py:654 at e805c6d), and the strace of the compile found no GPU-related path (hip.json, trace.gpu_paths).
- Pins: provenance.json carries the repository's toolchain pin files under pins (cuda, nvhpc, gcc, tt-metal, ttsim); this run used none of them. The framework versions are the ones it installed (below), and the ROCm compiler is the host's /opt/rocm/core-7.12/bin/hipcc, HIP 7.12.60610-2bd1678d3d (hip.json).

## Files

- batch.txt: the header: rx id, date, host, commit and tree state, Python, uv, the uv cache and scratch root sizes before the run, the core limit.
- stdout.txt: the batch's printed lines (40; the final status line is in the rx output). stderr.txt: the batch's own stderr, a harmless broken-pipe note of two lines from `sort | head` in the site-packages listing.
- requirements.in: the five pins asked for. resolved.txt: `uv pip compile --emit-index-annotation` of them, naming each package's index. install-tail.txt: the end of the install log. freeze.txt: `uv pip freeze` of the venv. freeze-check.json: the freeze against the expected pins and the resolution. pip-check.txt: `uv pip check`.
- sizes.txt: du of the venv, the uv cache before and after, and both together, with the hard-link count. site-packages-top.txt: the twelve largest site-packages entries.
- imports.json and imports-stdout.txt: the sandboxed import step: each module's version and import time, and torch's cuda, hip, git version, CPU capability, and one small matmul.
- hip.json and hip-stdout.txt: hipcc --version, the gfx942 compile, the inspection (which found no llvm-readelf in /opt/rocm/core-7.12/lib/llvm/bin), and the traced compile's summary.

## Results

The CPU framework set (stdout.txt, install-tail.txt, freeze-check.json, pip-check.txt):

- Installed with `uv pip install --no-config --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match --no-build` of torch==2.14.1+cpu, transformers==5.18.0, trl==1.14.1, peft==0.21.2, and accelerate==1.15.0: rc 0 in 6 s, "Resolved 59 packages in 773ms", "Prepared 50 packages in 3.83s", "Installed 59 packages in 396ms".
- 59 packages. The 11 expected pins all match: accelerate 1.15.0, datasets 5.1.0, huggingface-hub 1.33.0, numpy 2.2.6, pandas 2.3.3, peft 0.21.2, safetensors 0.8.0, tokenizers 0.23.2, torch 2.14.1+cpu, transformers 5.18.0, trl 1.14.1. The freeze equals the resolution; torch came from https://download.pytorch.org/whl/cpu and the other 58 from https://pypi.org/simple; `uv pip check` printed "All installed packages are compatible".

Sizes (sizes.txt, site-packages-top.txt):

| Measure | KiB |
| --- | --- |
| venv (du -sk) | 1174712 |
| venv, apparent size | 1117109 |
| uv cache before | 201824 |
| uv cache after | 1231436 |
| uv cache growth | 1029612 |
| cache and venv together | 1244224 |
| disk added (together, less the cache before) | 1042400 |
| site-packages/torch | 727236 |

24503 of the venv's 24734 files are hard links into the uv cache.

Imports in the sandbox (imports.json; 4 CPUs, 8192 MiB, 600 s, no network, the venv read-only): rc 0 in 10.74 s; all eleven modules imported at the versions above (torch 2.71 s, peft 4.24 s, datasets 1.24 s, transformers 0.96 s); torch.version.cuda null, torch.version.hip null, git_version 5c4886908584029761b579af026dcfb627c84070, cpu_capability AVX512, matmul_sum 83.0 as expected.

hipcc (hip.json, hip-stdout.txt), every step in the sandbox with /opt/rocm/core-7.12 read-only and only LANG=C, LC_ALL=C, PATH=/usr/sbin:/usr/bin:/sbin:/bin, and TMPDIR=/tmp set:

- `hipcc --version`: rc 0 in 0.48 s; "HIP version: 7.12.60610-2bd1678d3d", "AMD clang version 22.0.0git (https://github.com/ROCm/llvm-project.git c849bc16b0e49951d313756f20b73c2b28d321d7+PATCHED:9a6ac45c97a1e511db838c5b46257324d2de1780)", "Target: x86_64-unknown-linux-gnu".
- `hipcc --offload-arch=gfx942 -O2 -c probe.hip -o probe.o`: rc 0 in 1.84 s; probe.o 16952 bytes.
- `llvm-objdump --offloading probe.o`: "Extracting offload bundle: probe.o.0.host-x86_64-unknown-linux-gnu-" and "Extracting offload bundle: probe.o.0.hipv4-amdgcn-amd-amdhsa--gfx942" (hip.json, inspection.offloading).
- The same compile under `strace -f -e trace=%file`: rc 0 in 2.3 s, 4079 log lines. Programs started: /bin/sh x3, hipcc x1, clang++ x2, clang-22 x2, lld x1, clang-offload-bundler x1; no failed exec and no device-query program. /dev and /sys paths touched: /dev/null, /dev/urandom, /sys/devices/system/cpu/online; GPU-related paths: none.
- The batch's verdict line reads NO because /opt/rocm/core-7.12/lib/llvm/bin holds no llvm-readelf (hip.json, inspection.tools_missing; the batch looked only there, plans/spikes/p17-frameworks/frameworks.py:287, :298-300), so its section and code-object steps never ran, and its check of the llvm-objdump output looked for a line that tool does not print for this bundle. The object, pulled from the run's raw/ directory and read on the workstation, holds a gfx942 code object in its .hip_fatbin section (plans/spikes/p17-frameworks.md, Results 5); that reading is not part of this record.

## Left open

- The variables of /mnt/nvme10/john_ufl/rocm_env.sh: the file was unreadable ("Permission denied", rx 20261005-194413-exec-00bf), and this run set none of them.
- The CUDA and ROCm builds of torch were not installed; their sizes are read from index metadata in the spike report, not measured.
