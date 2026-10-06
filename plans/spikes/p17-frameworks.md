# Spike P17.1: framework pins, environments, and device assumptions

- Task: P17.1 (plans/p17-portable.md). Bible: Toolchain Pins, Repository Layout (placement rules), Training Module (Compute), Host Facts (ROCm), Risks And Questions (questions 6 and 15), Agent Rules 7 and 10.
- Status: complete. Results 1, 4, and 6 were read on the workstation from index metadata, Hub metadata, and upstream source, Results 2 from a local lock on the workstation, and Results 7 from the repository's own code at 1aa7b0f and e805c6d; Results 3 and 5 come from one batch on alpha01 from a clean commit, and Results 5 also from a workstation reading of the object that batch compiled.
- Labels: [MEASURED] marks only values from the batch, rx 20261005-204140-desktop-8r113ei-detached-e805c6d4-54d2, run from commit e805c6d on a clean tree (the rx record, results/p17-frameworks/provenance.json: dirty false, snapshot_of null). (metadata) marks a value read from a package index or the Hugging Face Hub; (source) one read from upstream source at a pinned tag, cited by the line of code that sets or compares it, never a comment; (workstation) a command run on the Windows workstation, which is exploratory, has no rx id, and is never [MEASURED].
- Dates: research on 2026-10-05 on the workstation (uv 0.12.17, Python 3.10.21). Read-only probes: rx 20261005-194413-exec-00bf, rx 20261005-203914-exec-99b8, and `rx doctor`, all 2026-10-05. Batch: 2026-10-05 20:41:40 to 20:42:07 on the alpha01 clock (UTC-07:00), 27 s, rc 0 (provenance.json start, end, rc).
- Base: branch p17-portable. Research at 1aa7b0f; the batch at e805c6d (e805c6d4af0a16e6458c4971362659b836112c50, "P17.1: Add the framework and hipcc spike batch"). `git diff --stat 1aa7b0f e805c6d -- lassi` is empty, so every lassi/ file:line in Results 7 holds at the spike's commit.
- Device: the batch ran on alpha01's CPU (nproc 256, Ubuntu 22.04.5 LTS; provenance.json host). No GPU device was opened: the imports and every hipcc step ran in the P0.16 sandbox, whose private /dev binds only null, zero, full, random, urandom, and tty (lassi/executors/sandbox.py:654 at e805c6d), and the gate's rocm_gpu class was disabled (`rx doctor`, 2026-10-05).
- Citations: "report/<file>" is results/p17-frameworks/<file>; "raw/<path>" is the run's raw/ directory, pulled into the untracked .rx/pulls/raw and never committed. Upstream source is cited as `<repo> <tag> <path>:<line>`. Full working notes (command transcripts, downloaded metadata, the lock experiment's files) stayed in the session's scratch directory; this report keeps the commands, the key output lines, and the sources.

## Question

What P17 builds on, as P17.1's seven acceptance bullets ask:

1. a PyTorch, Transformers, TRL, PEFT, and Accelerate set that installs together under Python 3.10 for the CPU, and the CUDA and ROCm builds that pair with it;
2. whether one uv lock can hold the CPU flavor beside the GPU flavors, or the GPU flavors need their own environment;
3. the measured size of the CPU set in a scratch venv;
4. the `/v1/models` and usage replies of vLLM, SGLang, and llama.cpp's server at pinned versions;
5. whether alpha01's hipcc builds gfx942 code without a GPU, and which variables its environment script sets (read, never sourced);
6. a small instruct model pinned by revision for gate part (a), with its size;
7. every place in lassi/ that assumes RNGD, Tenstorrent, or a device by name, with file:line.

Why it matters: P17.4 and P17.9 write the extras from 1 and 2 into pyproject.toml and uv.lock; P17.11 budgets scratch from 3 and 6 and fetches the model from 6; P17.3 builds its stub servers from 4; P17.6 stays READY or goes BLOCKED on 5; P17.2 to P17.7 take their seams from 7. Bible question 15 stays [OPEN] until 1 and 5 answer it.

Classification: factual. Every part is answered by index or Hub metadata, upstream source at a pinned tag, the repository's own code, or the one batch the plan allows. One part could not be answered: the environment script's variables (Results 5).

## Host state

- `rx doctor`, 2026-10-05 (doctor output, no rx id): scratch_free_gb 396 (the shared disk's free space), devices_enabled rocm_gpu false.
- The batch (report/batch.txt; provenance.json): du of the scratch root before it 101,570,812 KiB against the batch's stop line of 120,586,240 KiB (115 GiB); scratch_free_gb 395.9 before and 394.9 after; root_free_gb 244.6; mem_avail_gb 2950.6.
- rx 20261005-194413-exec-00bf (read-only): /opt/rocm/core-7.12/bin/hipcc is an ELF binary; HIP 7.12.60610-2bd1678d3d with AMD clang 22.0.0git; /opt/rocm/core-7.12/lib/llvm/bin holds amdgpu-arch and offload-arch, and /opt/rocm/core-7.12/bin holds rocm_agent_enumerator; reading /mnt/nvme10/john_ufl/rocm_env.sh gave "Permission denied".
- rx 20261005-203914-exec-99b8 (read-only): uv 0.12.17; /usr/bin/strace; /proc/sys/kernel/yama/ptrace_scope 1; util-linux 2.37.2.

## The batch run

Files, committed at e805c6d: plans/spikes/p17-frameworks/run.sh (the orchestrator), frameworks.py (the sandboxed steps and the summaries), and probe.hip (one `__global__` kernel with no host main and no runtime call). Added afterwards, untracked at e805c6d: inspect_bundle.py, a standard-library reader of the object's offload bundle (Results 5).

Command, from the clean commit in the detached clean worktree slot desktop-8r113ei-detached-e805c6d4 (run.sh header):

```
PYTHONIOENCODING=utf-8 uv run tools/rx.py run --timeout 7200 -- 'bash plans/spikes/p17-frameworks/run.sh'
```

Pulls: report/ into results/p17-frameworks/ with the command the batch's last line prints (`rx pull --path lassi-runs/p17-frameworks/<rx id>/report --into results/p17-frameworks`); the rx record's provenance.json beside it (AGENTS.md, Results); and raw/ into the untracked .rx/pulls/raw (`rx pull --path lassi-runs/p17-frameworks/<rx id>/raw`, inspect_bundle.py's docstring).

Steps, in order (run.sh header): 0. preflight and header: a core limit of exactly 1 byte (prlimit --core=1:1, checked with getrlimit), the gate's variables under /mnt/nvme10/joseph_ufl (Agent Rule 7), the scratch root under the 115 GiB stop line with room for a planned 4 GiB, then the date, commit, tree state, Python, uv, and the uv cache's du; 1. `uv venv --no-config --python 3.10 --no-managed-python --no-python-downloads` at build/p17-venv-cpu-<rx id> in the slot; 2. `uv pip compile --emit-index-annotation` with the install's index flags (the resolution record); 3. the install; 4. freeze against the expected pins, `uv pip check`, du of the venv, the cache, and both together; 5. the imports in the sandbox; 6. hipcc in the sandbox; 7. closing checks (the slot's tree still clean, du of the run directory, an ASCII pass over report/). The batch printed 41 lines: the 40 in report/stdout.txt and the exit trap's status line ("p17: ended with status 0 after 27 s"), then rx's own.

Batch shortcomings, neither of which changes a finding:

- The hip verdict line printed NO for two reasons; it needs rc 0 and either a code-object or an offloading match (frameworks.py:418). First, /opt/rocm/core-7.12/lib/llvm/bin holds no llvm-readelf (report/hip.json, inspection.tools_missing; the batch looked only in that directory, frameworks.py:287, :298-300), and frameworks.py's inspect_object opens the object with llvm-readelf first (frameworks.py:307), so the section list, the fatbin dump, the bundle list, and the code-object header steps never ran. Its llvm-objdump --offloading step did run in the sandbox and printed "Extracting offload bundle: probe.o.0.host-x86_64-unknown-linux-gnu-" and "Extracting offload bundle: probe.o.0.hipv4-amdgcn-amd-amdhsa--gfx942" (report/hip.json, inspection.offloading; raw/hip-inspect-work holds the extracted gfx942 entry, 5768 bytes). Second, the verdict counts that tool's output only on a "kind elf" line (frameworks.py:336-337), which it does not print for a clang offload bundle. report/hip-stdout.txt shows only the first line of that output. The object was read on the workstation instead (Results 5).
- report/stderr.txt holds two lines, "sort: write failed: 'standard output': Broken pipe" and "sort: write error", from `sort -rn | head -n 12` in the site-packages listing (run.sh:288); the listing's 12 lines are complete (report/site-packages-top.txt).

## Results

### 1. The framework set and the builds that pair with it (metadata, source)

Read on 2026-10-05 from PyPI's JSON API, the PyTorch index pages and their PEP 658 metadata files, and each project's build files at its release tag. Nothing here ran on alpha01.

| Package | Pin | Released (PyPI) | requires-python | torch bound it declares | Source line |
| --- | --- | --- | --- | --- | --- |
| torch | 2.14.1 | 2026-09-30 | >=3.10 | - | pytorch/pytorch v2.14.1 pyproject.toml:348 |
| transformers | 5.18.0 | 2026-09-30 | >=3.10.0 | torch>=2.5 (extra `torch`), and a runtime check that disables torch below 2.5.0 | huggingface/transformers v5.18.0 setup.py:51 (SUPPORTED_PYTHON_VERSIONS = (10, 14)), :319-320 (python_requires), :151 ("torch>=2.5"), :178 (extras["torch"]); src/transformers/utils/import_utils.py:198, :200 |
| trl | 1.14.1 | 2026-09-29 | >=3.10 | none; needs accelerate>=1.4.0, datasets>=4.7.0, transformers>=4.56.2 | huggingface/trl v1.14.1 pyproject.toml:30, :32, :33, :36 |
| peft | 0.21.2 | 2026-10-01 | >=3.10.0 | torch>=1.13.0; needs transformers and accelerate>=0.21.0 | huggingface/peft v0.21.2 setup.py:67, :73, :74, :76 |
| accelerate | 1.15.0 | 2026-09-09 | >=3.10.0 | torch>=2.0.0 | huggingface/accelerate v1.15.0 setup.py:79, :85 |

- Each is the newest release on PyPI on 2026-10-05, and each one's PyPI requires_dist matches its source lines. All accept Python 3.10. The highest torch floor is transformers' 2.5, and none sets an upper bound on torch. transformers also needs accelerate>=1.1.0 (setup.py:74), huggingface-hub>=1.31.0,<3.0 (:90), tokenizers>=0.23.1,<0.24.0 (:150), and safetensors>=0.8.0 (:136).
- Under Python 3.10, uv resolved datasets 5.1.0, huggingface-hub 1.33.0, tokenizers 0.23.2, safetensors 0.8.0, numpy 2.2.6, and pandas 2.3.3; Python 3.10 caps numpy at 2.2.x and pandas at 2.3.x. `uv pip compile` of the five pins for `--python-version 3.10 --python-platform x86_64-manylinux_2_28` resolved 59 packages (workstation), the same 59 names and versions alpha01 later installed (report/freeze.txt; compared on the workstation).
- Why the CPU index is needed: PyPI's torch 2.14.1 Linux wheel is the CUDA 13 build. Its requires_dist holds `cuda-toolkit[...]==13.0.3`, `nvidia-cudnn-cu13==9.24.0.43`, `nvidia-nccl-cu13==2.30.7`, and `triton~=3.8.0`, all gated on `platform_system == "Linux"`. The manylinux `+cpu` wheel's METADATA lists only filelock, typing-extensions>=4.10.0, setuptools>=77.0.3, sympy>=1.13.3, networkx>=2.5.1, jinja2, and fsspec>=0.8.5 (metadata). torch 2.14.1+cpu has a cp310 manylinux_2_28_x86_64 wheel of 196,197,995 bytes on https://download.pytorch.org/whl/cpu (metadata).
- Exploratory proof (workstation): the set installed into a fresh Python 3.10.21 venv on win_amd64 (60 packages in 26 s, 806M by `du -shl`); an import smoke test printed `torch 2.14.1+cpu`, `transformers 5.18.0`, `trl 1.14.1`, `peft 0.21.2`, `accelerate 1.15.0`, and `torch.version.cuda None hip None cuda.is_available False`; a tiny Llama built from a config and wrapped in LoRA through peft ran a forward pass, and trl's SFTConfig, DPOConfig, and GRPOConfig imported, with HF_HUB_OFFLINE=1 and no download.

CUDA builds of torch 2.14.1 (each index's torch/ page, newest stable cp310 x86_64 build; metadata):

| Index | Newest cp310 x86_64 | 2.14.1 cp310 files |
| --- | --- | --- |
| https://download.pytorch.org/whl/cu130 | 2.14.1+cu130 | manylinux_2_28_x86_64 (554,552,741 bytes), win_amd64 |
| whl/cu126 | 2.14.1+cu126 | manylinux_2_28_x86_64, win_amd64 |
| whl/cu132 | 2.14.1+cu132 | manylinux_2_28_x86_64, win_amd64 |
| whl/cu129, whl/cu128 | 2.13.0+cu129, 2.11.0+cu128 | none |

Pick: 2.14.1+cu130. Its METADATA requires `cuda-toolkit[...]==13.0.3`, the same CUDA as PyPI's default Linux wheel; cu126 requires cuda-toolkit 12.6.3 and nvidia-cudnn-cu12 9.10.2.21, and cu132 cuda-toolkit 13.2.2. There is no NVIDIA host (OQ-003), so which driver each build needs is not settled here.

ROCm builds of torch 2.14.1 (metadata):

| Index | Newest cp310 x86_64 | What it carries |
| --- | --- | --- |
| https://download.pytorch.org/whl/rocm7.2 | 2.14.1+rocm7.2, 6,224,803,911 bytes, uploaded 2026-09-30 | the ROCm 7.2 user space inside torch/lib |
| whl/rocm7.14 | 2.14.1+rocm7.14, 1,393,726,991 bytes | no ROCm libraries; pulls them from pip packages |
| whl/rocm7.1, whl/rocm7.0 | 2.13.0+rocm7.1, 2.10.0+rocm7.0 | - |

No rocm7.12 index exists; the index root lists rocm3.7 through rocm7.2 and rocm7.14.

- 2.14.1+rocm7.2: the index publishes no PEP 658 file for it, so it was read with HTTP range requests on the wheel's zip directory (under 3 MB of the 6.2 GB fetched). METADATA requires the seven pure-Python packages plus `triton-rocm~=3.8.0; platform_system == "Linux" and python_version < "3.15"`; triton_rocm 3.8.0 has a cp310 linux_x86_64 wheel of 361,431,641 bytes on the same index. The wheel's torch/version.py sets `hip = '7.2.53211'`, `rocm = '7.2.1'`, and git_version 5c4886908584029761b579af026dcfb627c84070. torch/lib holds 81 .so files, 7,125,155,359 bytes uncompressed, among them libamdhip64.so, libhsa-runtime64.so, libdrm.so and libdrm_amdgpu.so, libnuma.so, librocblas.so, libhipblaslt.so, libMIOpen.so, librccl.so, librocm_smi64.so, libamd_smi.so, and libtorch_hip.so, and gfx942 kernels (torch/lib/hipblaslt/library/Kernels.so-000-gfx942.hsaco, rocblas/library/Kernels.so-000-gfx942-xnack+ and -xnack-, hipsparselt/library/Kernels.so-000-gfx942.hsaco).
- 2.14.1+rocm7.14: METADATA requires `rocm[device-all,libraries]==7.14.*` and triton-rocm; torch/version.py sets hip 7.14.60850 and rocm 7.14.0, with the same git_version. `rocm` is an sdist only (rocm-7.14.1.tar.gz), linked to https://repo.amd.com/rocm/whl-multi-arch/; with rocm-sdk-core (414,575,399 bytes), rocm-sdk-libraries (142,984,638 bytes), and 25 rocm-sdk-device-gfx* wheels (rocm-sdk-device-gfx942 alone 1,601,733,459 bytes), its 27 ROCm files total 6,825,490,625 bytes, none with a sha256 on the index. PyPI holds unrelated placeholder projects under the names rocm, rocm-sdk-core, rocm-sdk-libraries, and rocm-sdk-device-gfx942, so every rocm name would need an explicit source.
- The rocm7.14 sdist probes GPUs when it is built (source): its requirements are dynamic (PKG-INFO `Dynamic: requires-dist`), its setup.py:51 calls `dist_info.determine_target_family()`, which reads ROCM_SDK_TARGET_FAMILY first (src/rocm_sdk/_dist_info.py:179) and otherwise runs `offload-arch` from PATH (_dist_info.py:137-138). Upstream offload-arch enumerates AMD GPUs from the KFD sysfs topology (llvm/llvm-project llvmorg-21.1.0 clang/tools/offload-arch/OffloadArch.cpp:44-51; AMDGPUArchByKFD.cpp:23-24, "/sys/devices/virtual/kfd/kfd/topology/nodes"). alpha01 has offload-arch in /opt/rocm/core-7.12/lib/llvm/bin (rx 20261005-194413-exec-00bf; AMD's build, source not read), so building that sdist there is a GPU probe (OQ-002). ROCM_SDK_TARGET_FAMILY=gfx942 avoids it (gfx942 is a valid family, _dist_info.py:406). The rocm7.2 flavor has no sdist and runs nothing at install.
- Neither wheel uses alpha01's ROCm 7.12: rocm7.2 carries its ROCm 7.2.1 user space in torch/lib, and rocm7.14 takes ROCm 7.14 from the rocm-sdk wheels in the venv. On a GPU, rocm7.2's bundled HSA thunk (source: ROCm/rocm-systems rocm-7.2.1 projects/rocr-runtime/libhsakmt/src/; fmm.c and topology.c read 2026-10-06) opens /dev/kfd (openclose.c:48, `kfd_device_name[] = "/dev/kfd"`, and :192, the open) and the render nodes (fmm.c:2357, `sprintf(path, "/dev/dri/renderD%d", minor)`, and :2358, the open), and reads the KFD sysfs topology (topology.c:56, `#define KFD_SYSFS_PATH "/sys/devices/virtual/kfd/kfd/topology"`; :63-65, get_topology_dir() returns it unless hsakmt_use_model is set; :781, the nodes directory). So it needs the amdgpu kernel driver and permission to open its nodes (OQ-040). rocm7.14's runtime was not read. Which amdgpu driver versions the 7.2.1 user space supports, and the full list of host paths it reads, were not read; both are P17.12 facts.

Pick: 2.14.1+rocm7.2. It is one hashed wheel plus triton-rocm, it is the ROCm index in uv's PyTorch guide (astral-sh/uv 0.12.17 docs/guides/integration/pytorch.md:113-114), it runs no build-time probe, and it ships gfx942 kernels. rocm7.14 is closer in version to the host's 7.12, but uses none of the host's user space and costs 28 more pyproject entries, unhashed files from a second host, and the offload-arch hazard.

### 2. One uv lock for the CPU and GPU flavors (workstation)

- Pattern: uv's documented one for PyTorch with optional dependencies (astral-sh/uv 0.12.17 docs/guides/integration/pytorch.md:381-440, extras plus `[tool.uv] conflicts` plus extra-scoped `[tool.uv.sources]` on `explicit = true` indexes; :207-227 for ROCm 7.2 on Linux).
- Setup: a scratch copy of the repository's pyproject.toml and uv.lock (the repository's files were never touched; their sha256 sums were unchanged after), with:

```toml
[project.optional-dependencies]
cpu  = ["torch==2.14.1", "transformers==5.18.0", "trl==1.14.1", "peft==0.21.2", "accelerate==1.15.0"]
cuda = [the same five]
rocm = [the same five, "triton-rocm==3.8.0 ; sys_platform == 'linux'"]

[tool.uv]
conflicts = [[{ extra = "cpu" }, { extra = "cuda" }, { extra = "rocm" }]]

[tool.uv.sources]
torch = [
    { index = "pytorch-cpu", extra = "cpu" },
    { index = "pytorch-cu130", extra = "cuda", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
    { index = "pytorch-rocm", extra = "rocm", marker = "sys_platform == 'linux'" },
]
triton-rocm = [{ index = "pytorch-rocm", extra = "rocm", marker = "sys_platform == 'linux'" }]

[[tool.uv.index]]   # three entries, each explicit = true
pytorch-cpu   = https://download.pytorch.org/whl/cpu
pytorch-cu130 = https://download.pytorch.org/whl/cu130
pytorch-rocm  = https://download.pytorch.org/whl/rocm7.2
```

- `uv lock` (empty cache, under a 3 GB cache watchdog) printed `Using CPython 3.10.21`, `Resolved 93 packages in 2.62s`, and `Added torch v2.14.1, v2.14.1+cpu, v2.14.1+cu130, v2.14.1+rocm7.2`, `Added triton v3.8.0`, `Added triton-rocm v3.8.0`; wall 3.1 s, exit 0. The lock grew from 20 packages (41,949 bytes) to 93 (219,030 bytes), and all 20 original packages kept their versions. Its five torch entries: 2.14.1 from the CPU index for darwin; 2.14.1+cpu from the CPU index for non-darwin platforms; 2.14.1+cu130 for Linux and win32, with cuda-bindings, cuda-toolkit, nvidia-cudnn-cu13, nvidia-cusparselt-cu13, nvidia-nccl-cu13, nvidia-nvshmem-cu13, and triton; 2.14.1+rocm7.2 for Linux, with triton-rocm; and PyPI's 2.14.1 as the fallback where no flavor build applies (outside Linux with the rocm extra; outside Linux and win32 with the cuda extra).
- What each extra installs, from `uv export --frozen --no-dev [--extra X] --no-hashes --no-emit-project --format requirements.txt` with markers evaluated for Python 3.10:

```
none  linux n=  9  (pyarrow, pyyaml, tiktoken and their deps: unchanged)
cpu   linux n= 63  torch==2.14.1+cpu
cpu   win   n= 64  torch==2.14.1+cpu
cuda  linux n= 82  torch==2.14.1+cu130 triton==3.8.0 cuda-toolkit==13.0.3.0 nvidia-cudnn-cu13==9.24.0.43 ... (18 CUDA packages)
cuda  win   n= 64  torch==2.14.1+cu130
rocm  linux n= 64  torch==2.14.1+rocm7.2 triton-rocm==3.8.0
rocm  win   n= 64  torch==2.14.1   (PyPI fallback; no ROCm build for Windows)
```

- `uv sync --frozen --extra cpu --extra cuda --no-install-project` failed: ``error: Extras `cpu` and `cuda` are incompatible with the declared conflicts: {`lassi[cpu]`, `lassi[cuda]`, `lassi[rocm]`}``. `uv sync --frozen --extra cpu --no-install-project` installed 68 packages in 6 s (the dev group included), and the venv printed `2.14.1+cpu 5.18.0 1.14.1 0.21.2 1.15.0 False`.
- Variant on whl/rocm7.14: uv refused extra-scoped sources for the 28 ROCm packages until each was also listed in the rocm extra (``Source entry for `rocm` only applies to extra `rocm`, but `rocm` was not found under the `project.optional-dependencies` section for that extra.``); with them listed, under ROCM_SDK_TARGET_FAMILY=gfx942, it locked 121 packages, the rocm sdist and the rocm-sdk wheels without hashes, in a pyproject.toml of 199 non-blank lines against 87 for rocm7.2.

Answer: yes, one uv lock holds the CPU flavor beside the CUDA and ROCm flavors, as mutually conflicting extras with extra-scoped torch sources on explicit indexes. A run picks one flavor (`uv sync --extra cpu`), uv refuses two at once, and a plain `uv sync` keeps the control plane's 9 packages on Linux. The GPU flavors need no environment definition of their own; each flavor is still installed into its own venv, since the conflict forbids mixing them. Only this layout, the five pins in each flavor extra, was tried.

### 3. The CPU set's size in a scratch venv [MEASURED]

Every value read from results/p17-frameworks/ in this subsection is [MEASURED 2026-10-05: rx 20261005-204140-desktop-8r113ei-detached-e805c6d4-54d2 from commit e805c6d (clean), alpha01's CPU; results/p17-frameworks/]. One run, so no value has a spread. Values from elsewhere carry their own source.

The install (run.sh:268-271), with uv 0.12.17 and the system Python 3.10.12, into a new venv at build/p17-venv-cpu-<rx id> in the slot:

```
uv pip install --no-config --python "$VENV/bin/python" --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match --no-build \
  "torch==2.14.1+cpu" "transformers==5.18.0" "trl==1.14.1" "peft==0.21.2" "accelerate==1.15.0"
```

uv's documentation treats the default index, `--index-url`, as lowest priority (astral-sh/uv 0.12.17 docs/concepts/indexes.md:365-367), so PyPI wins ties and only torch comes from the PyTorch index, as the resolution's annotations confirm (below); the `+cpu` local version admits only the CPU build. (workstation, exploratory, not [MEASURED]) `--index-strategy first-index` with the indexes the other way round was rejected: on the workstation it took 14 packages the PyTorch CPU index also hosts from that index, some at stale versions (certifi 2022.12.7, idna 3.4, packaging 24.1).

Output (report/stdout.txt, report/install-tail.txt, report/sizes.txt, report/freeze.txt, report/freeze-check.json, report/pip-check.txt):

```
resolve (uv pip compile, index annotations): rc=0; 59 pins
install: rc=0 in 6 s (1200 s limit); Resolved 59 packages in 773ms;Prepared 50 packages in 3.83s;Installed 59 packages in 396ms;
freeze: 59 packages; expected pins matching: 11 of 11
  indexes (uv pip compile annotations): https://download.pytorch.org/whl/cpu: 1 (torch), https://pypi.org/simple: 58
  freeze vs resolution: identical
pip check: rc=0; All installed packages are compatible
size: venv 1174712 KiB (apparent 1117109 KiB; 24503 of 24734 files hard-linked to the uv cache)
  uv cache 201824 -> 1231436 KiB; disk added (cache growth + venv-only blocks) 1042400 KiB
  largest site-packages entries (KiB): 727236=torch 154924=pyarrow 56592=transformers 45944=pandas 32384=numpy
```

| Measure | KiB | GiB |
| --- | --- | --- |
| venv, `du -sk` | 1,174,712 | 1.12 |
| venv, apparent size | 1,117,109 | 1.07 |
| uv cache growth (201,824 to 1,231,436) | 1,029,612 | 0.98 |
| disk added: du of cache and venv together, less the cache before | 1,042,400 | 0.99 |
| torch in site-packages | 727,236 | 0.69 |

- The 11 expected pins (the five, plus datasets 5.1.0, huggingface-hub 1.33.0, tokenizers 0.23.2, safetensors 0.8.0, numpy 2.2.6, pandas 2.3.3) all matched, and the freeze equals the resolution (report/freeze-check.json). install-tail.txt shows torch's download as 187.1 MiB.
- Imports, in the sandbox (4 CPUs, 8192 MiB, 600 s; the venv read-only; no network; HF_HUB_OFFLINE=1): rc 0 in 10.74 s; torch 2.14.1+cpu (import 2.71 s), transformers 5.18.0 (0.96 s), trl 1.14.1, peft 0.21.2 (4.24 s), accelerate 1.15.0, datasets 5.1.0 (1.24 s), huggingface_hub 1.33.0, tokenizers 0.23.2, safetensors 0.8.0, numpy 2.2.6, pandas 2.3.3; `torch.version.cuda=None torch.version.hip=None cpu_capability=AVX512`, and a 2x3 matmul summed to 83.0 as expected (report/imports.json, report/imports-stdout.txt).
- Against the plan's projection: the plan projected about 1 GB per environment for the CPU extras. The venv measures 1.12 GiB by du, and the install added 0.99 GiB of disk, since uv hard-links 24,503 of the venv's 24,734 files from its cache. A second CPU venv on the same cache would add little disk beyond its unlinked files (inferred from the link count, not measured). The scratch root stood at 101,570,812 KiB before the batch (report/batch.txt).

### 4. Served replies of vLLM, SGLang, and llama-server (source)

Read from source on 2026-10-05 at the tags below; no server was built or started. Every value is source-read, not measured.

| Server | Repo | Tag | Published (UTC) | Tag commit | Cited below as |
| --- | --- | --- | --- | --- | --- |
| vLLM | vllm-project/vllm | v0.31.0 | 2026-10-05T06:44:55Z | db9527a46873454610df6dbedf79a36d6bf1a7f6 | V, with E/ = vllm/entrypoints/ |
| SGLang | sgl-project/sglang | v0.5.21 | 2026-10-02T01:09:04Z | e00930c5489053f26d86b179cee0d087f846acbb | S, with E/ = python/sglang/srt/entrypoints/ |
| llama-server | ggml-org/llama.cpp | v0.6.0 | 2026-10-05T16:56:22Z | d81235049384534c167caea52b85a694f6103d14 | L, with T/ = tools/server/ |

Each tag is the project's newest stable release on 2026-10-05 (`gh api repos/<repo>/releases/latest`); for llama.cpp, whose build tags (b11433 and earlier) are marked prerelease, v0.6.0 is the only stable entry among the newest 100. The tags were checked out with `git clone --depth 1 --filter=blob:none --sparse --branch <tag>` and `git rev-parse HEAD` gave the commits above.

**vLLM v0.31.0.** GET /v1/models (V E/openai/models/api_router.py:20) returns `models_.model_dump()` (:25), so None fields are sent as null; there is no /v1/models/{id} route.

- List: `object` "list" (V E/serve/engine/protocol.py:117), `data` one card per served name, then one per LoRA adapter (:118; E/openai/models/serving.py:67-77, :152-165).
- Card: `id` the served name (protocol.py:106; serving.py:70); `object` "model" (:107); `created` int(time.time()) at each GET (:108); `owned_by` "vllm" (:109); `root` the `--model` path or hub id as given (:110; serving.py:72); `parent` null for the base (:111); `max_model_len` model_config.max_model_len for the base, null for LoRA (:112; serving.py:66, :71); `permission`, one ModelPermission whose `id` is "modelperm-" plus a new random id on each GET and whose `created` is now (protocol.py:91-102, :113).
- Served names are `--served-model-name`, else `[args.model]` (V E/launchers/api_server/app_state.py:49-52). A chat request naming another model gets 404 (V E/serve/engine/serving.py:49-50, :66-78) unless VLLM_SKIP_MODEL_NAME_VALIDATION is set (:76-77).
- Usage, non-streaming (V E/openai/chat_completion/serving.py): `prompt_tokens` = len(prompt_token_ids) (:1146-1148); `completion_tokens` = every generated token, reasoning included (:1149-1151, :1154); `total_tokens` their sum (:1155); `completion_tokens_details.reasoning_tokens` only with a reasoning parser (:1156-1160); `prompt_tokens_details` null unless `--enable-prompt-tokens-details` (:97-104; default False at E/launchers/cli_args.py:145). `system_fingerprint` is `vllm-<version>[-tpN][-ppN][-dpN][-ep]-<first 8 of the config hash>` (V E/serve/utils/fingerprint.py:58-84).
- Streaming: usage only with `stream_options.include_usage` or the server's `--enable-force-include-usage` (V E/serve/utils/api_utils.py:289-301), on a final chunk with `choices: []` (serving.py:878-887).
- Version: GET /version returns `{"version": vllm.__version__}` (V E/serve/instrumentator/basic.py:53-56). The `--api-key` check covers only paths starting with /v1, /v2, /inference, or /cohere (V E/serve/middleware/authenticate.py:11, :59), so /version needs no key.

**SGLang v0.5.21.** GET /v1/models (S E/http_server.py:1898) returns a ModelList; GET /v1/models/{model} (:1930-1952) answers 404 with code model_not_found for an unknown id (:1935-1946).

- Card: `id` tokenizer_manager.served_model_name (S E/openai/protocol.py:84; http_server.py:1901, :1908); `object` "model" (:85); `created` int(time.time()) at each GET (:86); `owned_by` "sglang" (:87); `root` the served name, not the path (:88; http_server.py:1909); `parent` null (:89); `max_model_len` tokenizer_manager.model_config.context_len (:90; http_server.py:1910); no `permission` field (protocol.py:81-90).
- The served name defaults to the model path (S python/sglang/srt/arg_groups/serving_hook.py:585-590), changes to the new path after /update_weights_from_disk (S python/sglang/srt/managers/tokenizer_manager.py:2231-2235), and may not hold ":" (S python/sglang/srt/arg_groups/validation_hook.py:131-136), since `base:adapter` in a request selects a LoRA adapter (S E/openai/serving_base.py:39-52). The chat reply echoes `request.model` (S E/openai/serving_chat.py:2449) with no check against the served name.
- Usage, non-streaming (S E/openai/usage_processor.py): `prompt_tokens` (:29-32), `completion_tokens` (:26-28), `total_tokens` (:123), and `reasoning_tokens` at the top level of usage (:35-37; protocol.py:215), with no completion_tokens_details; `prompt_tokens_details` only with `--enable-cache-report` and a cached count above 0 (:40-45). The reply has no `system_fingerprint` (protocol.py:1246-1254).
- Streaming: usage when `stream_options.include_usage` or the server's `--stream-response-default-include-usage` (S E/openai/utils.py:250-264), on a final chunk with `choices: []` (serving_chat.py:2221-2227).
- Version: there is no /version route. GET /server_info (S E/http_server.py:841-874) carries `version` (:867) beside every resolved ServerArgs field (:862) and `launch_command` (:863). `api_key` and `admin_api_key` are ServerArgs fields (S python/sglang/srt/arg_groups/fields/serving.py:153-160), and the source shows no step that drops them, so the reply carries any `--api-key` value in plain text (source-read, not measured). Every route but /health, /ready, and /metrics needs the key when one is set (S python/sglang/srt/utils/auth.py:101-106, :148-149).

**llama-server (llama.cpp v0.6.0).** GET /v1/models and GET /models share one handler (L T/server.cpp:268-269). Single-model mode (started with -m, -hf, or a docker repo) answers from L T/server-context.cpp:4922-4953; router mode (none of them, T/server.cpp:151-153) from T/server-models.cpp:2082-2143.

- Single-model list: a top-level Ollama-style `models` list of one entry (T/server-context.cpp:4926-4947), `object` "list" (:4948), and `data`, exactly one entry (:4949-4951).
- Entry: `id` meta.model_name (:4898); `aliases` and `tags` (:4899-4900); `object` "model" (:4901); `architecture` (:4902-4906); `created` std::time(0) at each GET (:4907); `owned_by` "llamacpp" (:4908); `meta` with vocab_type, n_vocab, `n_ctx` (the per-slot context), `n_ctx_train`, n_embd, n_params, size, and ftype (:4909-4918). No `root`, `parent`, `max_model_len`, or `permission` (:4894-4920).
- The model name is the first `--alias` in sorted order, else the -hf repo, the docker repo, or the -m path exactly as typed, else the file name (L T/server-context.cpp:1505-1514; common/common.h:312-320, :512).
- Router mode lists models that are not loaded, each with `status.value` downloading, downloaded, unloaded, loading, loaded, sleeping, or unknown (L T/server-models.cpp:2095-2111; T/server-models.h:54-60).
- Usage (L T/server-task.cpp:365-372): `completion_tokens` n_decoded, `prompt_tokens` n_prompt_tokens (the whole prompt, cached part included), `total_tokens`, and `prompt_tokens_details.cached_tokens` always. The reply's `model` is the server's own name whatever the request named (:444; T/server-context.cpp:4680), its `system_fingerprint` the build info (:445), plus a top-level `timings` object (:455-457). Streaming usage only with `stream_options.include_usage` (L T/server-schema.cpp:26-29; T/server-task.cpp:502-514), with no server flag to force it.
- Version: no /version route, and no reply carries the semantic version (set only in CMake, L CMakeLists.txt:6-9). GET /props (L T/server-context.cpp:5160-5170) carries `build_info` (:4989), `"b" + LLAMA_BUILD_NUMBER + "-" + LLAMA_COMMIT` (L common/build-info.cpp.in:27-30), the same string as every `system_fingerprint`. With `--api-key`, every path but /health, /v1/health, and the UI assets needs the key (L T/server-http.cpp:251-269).

Side by side:

| | vLLM v0.31.0 | SGLang v0.5.21 | llama-server v0.6.0 |
| --- | --- | --- | --- |
| list keys | object, data | object, data | models, object, data (single); data, object (router) |
| entry keys | id, object, created, owned_by, root, parent, max_model_len, permission | id, object, created, owned_by, root, parent, max_model_len | id, aliases, tags, object, architecture, created, owned_by, meta (+ status, source, can_remove in router mode) |
| context length | max_model_len | max_model_len (context_len) | meta.n_ctx (per slot), meta.n_ctx_train; /props n_ctx |
| fields new on each GET | created; permission[0].id and created | created | created |
| usage, non-streaming | prompt, completion, total; details null unless flags | prompt, completion, total, reasoning_tokens; prompt details null unless a flag | prompt, completion, total, prompt_tokens_details.cached_tokens always; top-level timings |
| checks the request's model | yes (404) | no | no (single-model mode answers with its own name) |
| version | GET /version, no key | GET /server_info `version`; key when set; the reply holds the keys | /props build_info `b<N>-<hash>`; key when set |

What lassi/llm reads today (at 1aa7b0f, unchanged at e805c6d): openai_compat's check() sends GET `<base_url>/models` (lassi/llm/openai_compat.py:102) and served_entry (lassi/llm/_http.py:294-312) needs a top-level `data` list of entries with a string `id` and returns the entry equal to the model id; complete() POSTs `/chat/completions` with `stream: false` (openai_compat.py:113-121) and reads `choices[0].message.content`, `usage.prompt_tokens`, and `usage.completion_tokens` (:123-125), each an int >= 0 (_http.py:286-291). The entry check() returns is recorded nowhere (the trial's model record is backend, model id, and sampling, lassi/llm/__init__.py:64-66). So the list parses and the two counts read in all three servers' non-streaming replies.

### 5. hipcc builds gfx942 code without a GPU; the environment script

The batch (frameworks.py hip): every step ran in the sandbox with the ROCm root /opt/rocm/core-7.12 exposed read-only, `env -i` giving exactly LANG=C, LC_ALL=C, PATH=/usr/sbin:/usr/bin:/sbin:/bin, and TMPDIR=/tmp (HIP_PATH and ROCM_PATH unset), and the compile sandbox's limits (report/hip.json, steps). [MEASURED 2026-10-05: rx 20261005-204140-desktop-8r113ei-detached-e805c6d4-54d2 from commit e805c6d (clean), alpha01's CPU, no GPU opened.]

```
hipcc: /opt/rocm/core-7.12/bin/hipcc (ROCm root /opt/rocm/core-7.12 -> /opt/rocm/core-7.12); env -i LANG LC_ALL PATH TMPDIR only
  hipcc --version: rc=0 in 0.48 s
    HIP version: 7.12.60610-2bd1678d3d
    AMD clang version 22.0.0git (https://github.com/ROCm/llvm-project.git c849bc16b0e49951d313756f20b73c2b28d321d7+PATCHED:9a6ac45c97a1e511db838c5b46257324d2de1780)
    Target: x86_64-unknown-linux-gnu
  compile (--offload-arch=gfx942 -O2 -c probe.hip -o probe.o): rc=0 in 1.84 s; probe.o 16952 bytes
  trace (strace -f -e trace=%file in the sandbox): rc=0 in 2.3 s; log 4079 lines
    programs started: /bin/sh x3, <rocm>/bin/hipcc x1, <rocm>/lib/llvm/bin/clang++ x2, <rocm>/lib/llvm/bin/clang-22 x2, <rocm>/lib/llvm/bin/lld x1,
  <rocm>/lib/llvm/bin/clang-offload-bundler x1
    failed exec attempts: 0; device-query programs: none; /dev and /sys paths: 3, GPU-related: none
```

(An excerpt of report/hip-stdout.txt: its sections, bundle targets, code object, objdump, and verdict lines are left out, and Batch shortcomings above covers them; the clang line whole from report/hip.json; `--version` also printed `InstalledDir: /opt/rocm/core-7.12/lib/llvm/bin`.)

- The traced compile ran `strace -f -qq -s 256 -e trace=%file -e signal=none` on the same compile to probe-trace.o. The three /dev and /sys paths it touched were /dev/null, /dev/urandom, and /sys/devices/system/cpu/online (report/hip.json, trace.dev_sys_paths). A grep of the pulled log (raw/hip-trace-work/trace.txt, workstation) for kfd, /dev/dri, /sys/class/drm, amdgpu-arch, offload-arch, rocm_agent_enumerator, and rocminfo matched only the four exec lines that carry the `--offload-arch=gfx942` argument.
- The object, read on the workstation from raw/hip-compile-work/probe.o (sha256 f67eb04a2600d8f4f9fbcbda37969c1c3fce3f9f1565388908eb74f55d4930d3):

```
$ uv run python plans/spikes/p17-frameworks/inspect_bundle.py .rx/pulls/raw/hip-compile-work/probe.o
.hip_fatbin: offset 0x1000, 9864 bytes
entry host-x86_64-unknown-linux-gnu-: 0 bytes, not an ELF (host entries are empty)
entry hipv4-amdgcn-amd-amdhsa--gfx942: 5768 bytes, ELF e_machine 224, e_flags 0x54c, gfx942 code: True
$ wsl readelf -S -W .rx/pulls/raw/hip-compile-work/probe.o      (the HIP lines)
  [ 7] .hip_fatbin       PROGBITS        0000000000000000 001000 002688 00   A  0   0 4096
  [ 8] .hipFatBinSegment PROGBITS        0000000000000000 003688 000018 00  WA  0   0  8
  [ 9] .rela.hipFatBinSegment RELA            0000000000000000 003b08 000018 18   I 18   8  8
```

  The traced compile's raw/hip-trace-work/probe-trace.o gave the same three inspect_bundle lines. e_machine 224 is EM_AMDGPU, and the low byte of e_flags, 0x4c, is EF_AMDGPU_MACH_AMDGCN_GFX942 (source: llvm/llvm-project llvmorg-21.1.0 llvm/include/llvm/BinaryFormat/ELF.h:321, :759, :841; upstream LLVM, not AMD's build). The batch's own llvm-objdump --offloading step extracted the same 5768-byte gfx942 entry in the sandbox (Batch shortcomings above).
- The environment script: /mnt/nvme10/john_ufl/rocm_env.sh gave "Permission denied" (rx 20261005-194413-exec-00bf), so the variables it sets could not be read. The compile above needed none of them: it ran with only LANG, LC_ALL, PATH, and TMPDIR set, hipcc invoked by its absolute path.
- Not tried: a compile without --offload-arch (the tree holds amdgpu-arch, offload-arch, and rocm_agent_enumerator, rx 20261005-194413-exec-00bf, which the batch treats as device-query programs, frameworks.py:64, and never ran; upstream offload-arch, also run as amdgpu-arch, enumerates GPUs, llvm/llvm-project llvmorg-21.1.0 clang/tools/offload-arch/OffloadArch.cpp:44-51, :72-73; rocm_agent_enumerator's source was not read); a compile that fails, for diagnostics fixtures; linking a program.

Answer: yes. alpha01's hipcc builds gfx942 device code without a GPU, compile-only with an explicit --offload-arch=gfx942, inside the sandbox whose private /dev holds no GPU node, with no ROCm variable set, starting no device-query program and touching no GPU path. Which variables rocm_env.sh sets is not known: the file is unreadable.

### 6. The gate model (metadata; workstation counts)

Pin: Qwen/Qwen2.5-Coder-0.5B-Instruct at revision ea3f2471cf1b1f0db85067f1ef93848e38e88c25 (Hub API sha; lastModified 2024-11-18T12:52:41Z).

| Key | Value |
| --- | --- |
| weights | model.safetensors, 988,097,824 bytes (0.92 GiB), lfs sha256 f9523886352217ded3aeeef552b381af79d568c6d49a4b9e423288cea56b0a44; 494,032,768 BF16 parameters (API safetensors.total) |
| repository at the revision | 10 files, 999,604,233 bytes, so a full snapshot fetch is about 1.0 GB, with no file filter needed |
| license | Apache-2.0 (API cardData.license, tag license:apache-2.0; LICENSE file); not gated |
| context | 32768 (config.json:12 max_position_embeddings; tokenizer_config.json:202 model_max_length agrees) |
| architecture | Qwen2ForCausalLM (config.json:3), model_type qwen2 (:14); 24 layers, hidden 896, 14 attention heads, 2 KV heads, tied embeddings |
| chat template | present (tokenizer_config.json:198), ChatML form with a system message; eos `<\|im_end\|>` |
| Transformers | >= 4.37.0, the first release that maps qwen2 (huggingface/transformers v4.37.0 src/transformers/models/auto/configuration_auto.py:185; absent at v4.36.2); 5.18.0 maps it (v5.18.0 auto_mappings.py:532, modeling_auto.py:851), and AutoConfig under 5.18.0 parsed the config as Qwen2Config, max_position_embeddings 32768, dtype bfloat16 (workstation) |
| file sha256 at the revision | config.json b1e58593cd31852f7da5c2fc31ddf6135b9c066c0fd9177a4bbe95717083adff; tokenizer_config.json 959e7f1d9a1b7641a6d6ce05ca97b75c7894fcb66cbe5a040406458fb1128ee4; generation_config.json fdaccbcb02f3e1e7914ccb0f69ebe899071ffd27cf825166d823163e156870f2; tokenizer.json c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539 |

Sources: `curl https://huggingface.co/api/models/<id>`, `.../revision/main?blobs=true`, and `curl -L https://huggingface.co/<id>/resolve/<sha>/<file>` for config.json, tokenizer_config.json, generation_config.json, and tokenizer.json (no weights downloaded).

The longest prompt the recipe sends (workstation, exploratory). Gate part (a)'s recipe is modeled on projects/lassi-demo/rngd-cpu.yaml, which extends rngd-compile.yaml: prompts lassi-2024 and context [openmp-4.0-card, cuda-12.5-ch5] (rngd-compile.yaml:29-30), max_tokens 4096 (:22), max_corrections 5 (:16), parsed_diagnostics off (:13). Every call sends one system and one user message (lassi/core/stages.py:802). The prompts were built with the repository's own functions from the locally generated prompt set and packs and the 20 item sources (third_party/LASSI at 74b4681, each checked against its sha256 in assets/bench/lassi-hecbench-10.yaml), and counted with the chosen model's tokenizer under Transformers 5.18.0 as apply_chat_template([system, user], add_generation_prompt=True):

- The packs are 7,303 tokens for omp and 3,932 for cuda. Over the ten items, the fixed part of the first generate prompt (empty summary and description) is 4,864 to 6,038 tokens for omp to cuda and 8,336 to 9,775 for cuda to omp, the largest bsearch cuda to omp.
- The generate call's bound: 9,775 + 2 x 4,096 (summary and description replies) + 4,096 (its own reply) = 22,063 tokens, leaving 10,705 of the 32768 context.
- Correction prompts have no bound that fits any context. One holds 159 tokens of fragments, the previous target file (at most one 4,096-token reply), and the error text, with 4,096 more for the reply, which leaves 24,417 tokens for the error text. With parsed_diagnostics off, the compile error text is the raw compiler stderr (stages.py:1341), kept up to 64 MiB (lassi/executors/sandbox.py:401, COMPILE_OUTPUT_CAP_BYTES); a run error carries the program's stderr (stages.py:1556), kept up to 1 MiB (lassi/toolchains/_base.py:60, OUTPUT_CAP_BYTES) in any setting. parsed_diagnostics on caps the compile text at 50 diagnostics and 16,384 bytes (stages.py:328-329).

Candidates (metadata; counts on the workstation):

| Model | Revision | Weights | Context | Verdict |
| --- | --- | --- | --- | --- |
| Qwen/Qwen2.5-Coder-0.5B-Instruct | ea3f2471cf1b1f0db85067f1ef93848e38e88c25 | 988,097,824 B | 32768 | chosen |
| Qwen/Qwen2.5-0.5B-Instruct | 7ae557604adf67be50417f59c2c2f167def9a775 | 988,097,824 B | 32768 (its tokenizer_config says 131072) | fits; equal alternate |
| Qwen/Qwen3-0.6B | c1899de289a04d12100db370d81485cdf75e47ca | 1,503,300,328 B | 40960 | fits; larger, needs Transformers >= 4.51.0, thinks by default |
| HuggingFaceTB/SmolLM2-360M-Instruct | a10cc1512eabd3dde888204e902eca88bddb4951 | 723,674,912 B (repository 5,024,415,389 B with onnx files) | 8192 | refused: its cuda to omp summarize prompt alone is 8,304 tokens |

All four are Apache-2.0. The two Qwen2.5 0.5B models share size, architecture, tokenizer (the same tokenizer.json sha256), license, and context; the Coder model is chosen because its tokenizer's model_max_length agrees with its config and because it is tuned on code. Quality is not a criterion. Qwen3-0.6B's template leaves thinking on unless the caller passes enable_thinking=False (its tokenizer_config.json:230), which a generic hf_local would have to pass.

PROJECTED, derived from config.json and the API's parameter count, not measured: fp32 weights are 494,032,768 x 4 B, about 1.98 GB; the KV cache is 24 layers x 2 x 2 KV heads x 64 dims x 4 B = 24,576 B per token, about 0.54 GB at 22,063 tokens. CPU generation speed was not measured.

The model's generation_config.json sets do_sample true, repetition_penalty 1.05, temperature 0.7, top_p 0.8, and top_k 20 (generation_config.json:4, :9-12), which generate() applies unless the call overrides them; the recipe sets only temperature 0.2 and top_p 0.9 (projects/base.yaml:4-5).

### 7. Device assumptions in lassi/ (file:line at e805c6d)

Classes: (a) a target or vendor name inside its own adapter, correct by placement (Design Principle 9); (b) a device assumption in device-neutral or shared code; (c) a provenance or record field that names or omits a device.

Searches (`git grep` at 1aa7b0f, whose lassi/ equals e805c6d's):

```
P='rngd|furiosa|\bnpus?[0-9]*\b|tenstorrent|ttsim|wormhole|blackhole|tt-metal|tt_metal|ttmetal|tt-mlir|tt_mlir|cuda|nvcc|nvhpc|nvc\+\+|nvidia|nvml|\bhip\b|hipcc|rocm|\bamd\b|mi300|gfx9|gpus?\b|\bcpus?\b|device|8123'
git grep -n -i -E "$P" 1aa7b0f -- lassi | wc -l        -> 588 (37 files)
```

Lines per term: rngd 0, furiosa 5, npu 0, tenstorrent 0, ttsim 80, wormhole 7, tt-metal 114, tt-mlir 2, cuda 51, nvcc 54, nvhpc 14, nvidia 0, nvml 0, hip and hipcc 0, rocm 2, amd 0, mi300 0, gfx9 0, gpu 18, cpu 93, device 183, 8123 2. Nothing in lassi/ names RNGD, an NPU, NVML, HIP, AMD, the MI300X, or gfx9; the only rocm hits are the ToolchainPins field (lassi/core/record.py:38, :464). By area: lassi/toolchains 234, lassi/executors 197, lassi/core 101, lassi/present 19, lassi/analysis 15, lassi/scoring 13, lassi/llm 4, lassi/bench 3, lassi/cli.py 2. Value searches (default URLs and ports, device strings, architecture presets, device files, provenance fields) and structure searches (device(), the simulator capability, the pin-field refusal, the refused keys, how the backend is built) followed, and each hit was read.

A local check of the pin-field refusal (workstation; `PYTHONPATH=c:/dev/lassi` with the repository's venv): `_pin_versions` accepted `rocm` and refused `hipcc` and `torch` with "RunError: the pin 'hipcc' has no Trial toolchain_pins field; fields: ('llvm', 'polygeist', 'tt_mlir', 'tt_metal', 'ttsim', 'furiosa_sdk', 'cuda', 'nvhpc', 'rocm', 'gcc')".

| # | file:line | What it assumes or omits | Class | Task |
| --- | --- | --- | --- | --- |
| 1 | lassi/core/runner.py:1965 (_provenance) | `"driver": None` for every run: no driver or runtime version in provenance.json | c | P17.2 |
| 2 | lassi/core/runner.py:1992, :1997 (_trial_provenance) | a trial copies one device string (its target language's) and takes `sdk` from the null driver; no place for a device record | c | P17.2 |
| 3 | lassi/core/record.py:473-494 (Provenance) | `device` and `sdk` are one string each; no field for the model's device or the framework | c | P17.2 |
| 4 | lassi/core/record.py:37-39 (TOOLCHAIN_PIN_NAMES), :453-465 (ToolchainPins); lassi/core/runner.py:1705-1708 (_pin_versions) | a closed set of pin fields; any other name is refused (hipcc and torch above); no slot for framework pins; the same set drives the Parquet columns (lassi/core/parquet.py:93, :278) and trial.md's pin table (lassi/core/trial_md.py:211-212) | c | P17.2 and P17.4 (framework pins); P17.6 (the hipcc pin's name) |
| 5 | lassi/core/parquet.py:96-97, :288-289 | string columns provenance_device and provenance_sdk; none for a device record or framework | c | P17.2 |
| 6 | lassi/core/record.py:524-529 (ModelInfo); lassi/core/recipe.py:406 | a model arm is backend, id, and sampling: no revision, device, server URL, or framework version | c | P17.2, P17.3, P17.4 |
| 7 | lassi/core/interfaces.py:146-151 (LLMBackend) | no way for a backend to name its device, unlike Executor's device() (:199-221) | b | P17.2, P17.4 |
| 8 | lassi/core/runner.py:513 (`factory(settings.model_id)`) with lassi/llm/openai_compat.py:41 (`DEFAULT_BASE_URL = "http://127.0.0.1:8123/v1"`) | every openai_compat run reaches port 8123, the furiosa-llm port of Agent Rule 8; no other server can be named | b | P17.3 |
| 9 | lassi/core/runner.py:292 (`_NOT_CARRIED_OUT`) | `profiler` is refused, so Attempt.profile is never filled | b | P17.7 |
| 10 | lassi/core/record.py:345-350 (Profile); lassi/core/interfaces.py:234-243 (Profiler) | runtime_s, avg_power_w, and energy_j carry no profiler name or device, and the Profiler has no device() | c | P17.7 |
| 11 | lassi/core/runner.py:1939 (_device) | for an executor without device(), the runner writes its own "none (compile only)", a copy of lassi/executors/none.py:18 | c | P17.2 |
| 12 | lassi/core/runner.py:2086-2092, :2113; lassi/scoring/score_run.py:128, :653-667 | run.md and review.md show the device string and a Driver row that is always "-" | c | P17.2 |
| 13 | lassi/analysis/metrics.py:386; lassi/analysis/tables.py:148; lassi/present/live.py:351, :487 | tables and the live header list provenance.device; correct today, they follow P17.2's type | c | P17.2 (follow-on) |
| 14 | lassi/core/stages.py:915-916, :949 with lassi/core/runner.py:1992 | under baseline_both the source reference runs on the source language's executor, but the trial names only the target's device; provenance.json's `devices` has both | c | none in P17 |
| 15 | lassi/core/record.py:732-738 (arm_segment) with lassi/analysis/metrics.py:404-420 | the arm is the model id alone, so one model on cpu and on rocm shares an arm | c | none in P17 (each gate part is its own run) |
| 16 | lassi/core/stages.py:313-316 (HANG_DIAGNOSTIC); lassi/core/capabilities.py:44 (WATCHER_CODE) | the hang text names tt-metal constructs and goes to any `simulator` executor | b | none (the gpu executor is not a simulator) |
| 17 | lassi/bench/registry.py:104, :182, :377-379 | an optional item key for a tt-metal compute-config note in the shared registry | b | none |
| 18 | lassi/executors/sandbox.py:654 (six /dev nodes), :695-697 (/sys hidden), :353-363 (TT_METAL_NAMES), :372-374 (ENVIRONMENT_NAMES) | no GPU node, no GPU /sys path, and no visibility variable; placement is correct | a | P17.5 |
| 19 | lassi/toolchains/pins.py:47 (`PREFIX_VARIABLES = {"CUDA_HOME_FROM": "NVHPC_CUDA_HOME"}`) | the only pin-to-compile variable; any variable hipcc needs is not yet allowed | a | P17.6 |
| 20 | lassi/executors/native.py:45-46, :192 | device() names the CPU from /proc/cpuinfo ("host CPU (native): <model>") | a | P17.2 (a cpu-kind probe should name the CPU the same way) |

Read as correct and left out: ttsim's device() names the simulator (lassi/executors/ttsim.py:562-573); PROXY_CAPABILITIES and openmp_multicore decide by capability (lassi/core/recipe.py:80-82, :884-899); REFERENCE_CPUS and RUN_CPUS (stages.py:335, :351) and Limits (interfaces.py:78-83) are host CPU limits; ollama's port 11434 (lassi/llm/ollama.py:34) is Ollama's own default. No registry name (native, none, ttsim, nvcc-sm80, nvcpp-cc80, gcc-native, ttmetal-host, openai_compat, ollama) is a default in lassi/core, lassi/cli.py, lassi/analysis, lassi/scoring, lassi/present, or lassi/bench, and projects/base.yaml names no executor, toolchain, model, or device.

Adapter-local hits (class a), by file: lassi/executors ttsim.py 96 (`ARCH = "wormhole_b0"` at :125), sandbox.py 69, native.py 17, __init__.py 10, none.py 5; lassi/toolchains nvcpp.py 48 (`GPU = "cc80"` at :249), ttmetal_guard.py 44, nvcc.py 40 (`ARCH = "sm_80"` at :180), ttmetal_build.py 24, _cxx_flow.py 23, __init__.py 18, _stderr.py 16, pins.py 11, _base.py 6, gcc.py 2, _cxx_scan.py 2. They are counted here, not listed one by one: the search command above, run at 1aa7b0f or e805c6d, lists every one with its file:line.

Hits that name no device: "null device" (lassi/cli.py:42, :157; lassi/core/progress.py:38, :127, :142, :162; lassi/present/live.py:68); st_dev (lassi/present/settings.py:90-162); CUDA as a source language or direction (lassi/core/files.py:14, :67-68, :91; lassi/core/fragments.py:25; lassi/llm/mock.py:15; lassi/scoring/similarity.py:13, :43, :152, :167; lassi/analysis/paper.py:13; lassi/bench/registry.py:121); docstring examples (lassi/core/record.py:735; lassi/present/live.py:5; lassi/llm/openai_compat.py:3-5). lassi/core/files.py's suffix table has no `.hip` entry.

## Finding

1. torch 2.14.1, transformers 5.18.0, trl 1.14.1, peft 0.21.2, and accelerate 1.15.0 install together under Python 3.10 on the CPU: 59 packages, a venv of 1,174,712 KiB (1.12 GiB) that added 1,042,400 KiB of disk, imported in the sandbox with torch 2.14.1+cpu, no CUDA, and no HIP [MEASURED, rx 20261005-204140-desktop-8r113ei-detached-e805c6d4-54d2]. The builds that pair with it are 2.14.1+cpu from whl/cpu, 2.14.1+cu130 from whl/cu130, and 2.14.1+rocm7.2 from whl/rocm7.2 with triton-rocm 3.8.0 (metadata).
2. One uv lock holds all three flavors as conflicting extras with extra-scoped torch sources; uv refuses two flavors at once, and a plain `uv sync` keeps the control plane (workstation).
3. The three servers' `/v1/models` lists all parse with LASSI's served_entry, and the usage counts it reads are present in all three, but only vLLM and SGLang send max_model_len, every entry has fields that change on each GET, the version lives in a different place in each, and SGLang's /server_info reply carries the API keys (source).
4. alpha01's hipcc builds gfx942 device code compile-only with an explicit --offload-arch, with no GPU device in view and no ROCm variable set, and llvm-objdump --offloading extracted the object's gfx942 entry [MEASURED, the same rx id]; read on the workstation, that entry is gfx942 code (Results 5). rocm_env.sh is unreadable, so its variables are unknown.
5. The gate model is Qwen/Qwen2.5-Coder-0.5B-Instruct at ea3f2471cf1b1f0db85067f1ef93848e38e88c25: 988,097,824 bytes of weights, about 1.0 GB to fetch, a 32768 context that holds the recipe's longest bounded call of 22,063 tokens (metadata; workstation counts).
6. lassi/ names no RNGD, NVML, HIP, AMD, or MI300X; the device assumptions P17 acts on are the provenance and record fields of rows 1 to 6 and 10 to 12, the backend construction of row 8, and the missing seams of rows 7 and 9.

Confidence: high for the CPU install, size, and imports (one clean run, the freeze equal to the resolution, which also equals the workstation's); high for the hipcc answer (rc 0; the bundle's gfx942 entry read on the workstation, its sections listed by readelf, and the same entry extracted by llvm-objdump in the sandbox; a trace with no device-query program or GPU path), though one small kernel was compiled and no failing compile or link was tried; high for the pins and the lock pattern, which come from metadata, source, and a local lock and sync, but the CUDA and ROCm builds were never installed; medium for the server replies, which are source-read, never served; high for the model's metadata and the bounded prompt count, none for its CPU speed; high for the sweep's rows, which were read at each hit, though a name search can miss an assumption no pattern names.

## Consequences for the plan

Firm from the evidence:

- P17.2: the seams are rows 1 to 7 and 11 to 13 of Results 7. A cpu-kind probe should name the CPU as the native executor does (row 20), so the model's device and an executor's device compare equal. Framework versions have no Trial.toolchain_pins field (row 4); they go with the device record (the plan's "framework and version"), unless a task adds pin fields with a Result Record edit. Fake probes can take their values from the measured CPU build (torch.version.cuda None, hip None) and the rocm7.2 wheel's torch/version.py (hip 7.2.53211, rocm 7.2.1).
- P17.3: the stub servers answer in Results 4's shapes. The record must allow max_model_len to be absent (llama-server sends meta.n_ctx instead); `created`, and vLLM's permission[0].id and permission[0].created, change on every GET and stay out of any hash or comparison; version endpoints sit at the server root, outside base_url's /v1, and every one but vLLM's /version needs the key when one is set; from SGLang's /server_info only `version` is read, and the reply is never stored or logged (Agent Rule 12). check() stays required before the first request, since SGLang and llama-server do not check the request's model. A llama-server recipe needs `--alias` for a stable id that holds no host path; router-mode llama-server lists models that are not loaded, so either refuse it or require `status.value == "loaded"`; an SGLang id cannot hold ":", and its id changes after /update_weights_from_disk.
- P17.4: the extras follow Results 2's layout; splitting the five pins into an hf extra and a train extra beside a flavor extra was not tried and needs its own `uv lock` check. hf_local reads the context from the config's max_position_embeddings, not the tokenizer's model_max_length (the two disagree for Qwen2.5-0.5B-Instruct), and passes every sampling field, since the model's generation_config sets defaults that generate() otherwise applies. Correction prompts can pass the context (Results 6), so how the runner ends a trial whose request hf_local refuses decides whether gate part (a) can fail on one. Imports cost seconds (torch 2.71 s, peft 4.24 s in the sandbox), which the registration rule's lazy import already covers. Never pass uv's `--torch-backend=auto` on alpha01: uv's guide says it queries the installed CUDA driver and AMD GPU versions (docs/guides/integration/pytorch.md:514-516), a device probe under OQ-002.
- P17.5: row 18 is already in its acceptance. For the AMD table, the rocm7.2 runtime's source names /dev/kfd, the /dev/dri render nodes, and the KFD topology under /sys/devices/virtual/kfd/kfd/topology (Results 1), which the sandbox now hides with every /sys/devices entry but system (lassi/executors/sandbox.py:695-697); P17.12 measures the full list.
- P17.6: hipcc needs no GPU, so the task stays READY. EXECUTABLE /opt/rocm/core-7.12/bin/hipcc with the banner line `HIP version: 7.12.60610-2bd1678d3d`; always an explicit `--offload-arch=gfx942`, never amdgpu-arch, offload-arch, or rocm_agent_enumerator; the measured compile needed only the sandbox's LANG, LC_ALL, PATH, and TMPDIR, with the ROCm root read-only, so PREFIX_VARIABLES (row 19) may need no new name. The pin cannot be named hipcc without a Result Record edit (row 4); `rocm` is an existing field. rocm_env.sh is unreadable, so its variables cannot go into the pin, and the Host Facts reconciliation rests on the measured compile. /opt/rocm/core-7.12/lib/llvm/bin holds no llvm-readelf (report/hip.json, inspection.tools_missing): remote tests and fixture captures check objects with llvm-objdump --offloading, which is there, or a standard-library reader such as inspect_bundle.py. Its Execution Backends edit also updates the gpu (AMD) row, which still makes the compile-only tier wait on "if hipcc builds gfx942 code without a GPU (P17 spike)" (the bible's Execution Backends table, gpu (AMD) row), now answered. No diagnostics fixture was captured: the probe compiled clean. lassi/core/files.py has no `.hip` suffix (Results 7, hits that name no device); check whether the toolchain's sources need one.
- P17.7: rows 9 and 10.
- P17.8: nothing new from the spike.
- P17.9: the train extra's pins are Results 1's; trl's SFTConfig, DPOConfig, and GRPOConfig import with them (workstation, exploratory). The bible's Training Module Episodes paragraph still cites TRL 1.9.2's multi-turn report (the bible's Training Module, Episodes) while the pin is now 1.14.1; recheck it there.
- P17.10: on a Linux runner the cpu extra pulls no CUDA package (Results 2, uv export), and the set downloads torch at 187.1 MiB (report/install-tail.txt).
- P17.11: the model fetch is about 1.0 GB at the pinned revision, with no file filter; fp32 weights need about 1.98 GB of memory and the KV cache about 0.54 GB at 22,063 tokens (PROJECTED, derived, Results 6); the CPU set added 0.99 GiB of scratch to a uv cache that lacked it (measured). p17-cpu-hf.yaml can turn parsed_diagnostics on to cap compile error text at 16,384 bytes; a run's stderr stays capped only at 1 MiB.

For the GPU tasks (P17.12, P17.13), from metadata: the rocm7.2 wheel is a 6,224,803,911-byte download whose torch/lib unpacks to 7,125,155,359 bytes, so a ROCm venv is PROJECTED at about 7 to 8 GB of scratch, from the wheel's uncompressed torch/lib (not measured; check du against the 115G stop line first). It brings its own ROCm 7.2.1 user space; on a GPU its runtime opens /dev/kfd and the render nodes and reads the KFD sysfs topology (Results 1, from source), so it needs the amdgpu kernel driver and the render group grant (OQ-040). Which driver versions it supports, and the full list of host paths it reads, are P17.12 facts. Never install the rocm7.14 flavor on alpha01 without ROCM_SDK_TARGET_FAMILY=gfx942. Which driver cu130 needs waits for an NVIDIA host (OQ-003). Bible question 6 (ROCm builds of TRL and vLLM on gfx942) stays open: the rocm7.2 wheel ships gfx942 kernels, but nothing ran on a GPU and vLLM's ROCm build was not examined; P17.1's bible edit leaves it as it is, so P17.12 or P17.13 records its answer.

## Proposed bible edit

Applied by the main session to the master and the mirror, with the Decision Log count re-read at apply time:

- Toolchain Pins: the framework pins (torch 2.14.1 with its cpu, cu130, and rocm7.2 builds and indexes, transformers 5.18.0, trl 1.14.1, peft 0.21.2, accelerate 1.15.0) as optional extras, and the gate model pin.
- Host Facts, the ROCm bullet: rocm_env.sh unreadable on 2026-10-05; hipcc compile-only for gfx942 with an explicit --offload-arch [MEASURED: rx 20261005-204140-desktop-8r113ei-detached-e805c6d4-54d2]; the tree's device-query programs, and no llvm-readelf in its lib/llvm/bin; the ROCm torch wheel's own ROCm 7.2.1 user space and what its runtime opens on the host (source).
- Risks And Questions: question 15 answered.
- Decision Log: one row dated 2026-10-05, and the count sentence raised by one.

## Sources

- alpha01: `rx doctor` (2026-10-05); rx 20261005-194413-exec-00bf and rx 20261005-203914-exec-99b8 (read-only); rx 20261005-204140-desktop-8r113ei-detached-e805c6d4-54d2 from commit e805c6d, pulled into results/p17-frameworks/ (provenance.json, batch.txt, stdout.txt, stderr.txt, requirements.in, resolved.txt, install-tail.txt, freeze.txt, freeze-check.json, pip-check.txt, sizes.txt, site-packages-top.txt, imports.json, imports-stdout.txt, hip.json, hip-stdout.txt) and the untracked .rx/pulls/raw (probe.o, probe-trace.o, trace.txt, the extracted bundle entries).
- Package indexes, read 2026-10-05: https://pypi.org/pypi/<package>/json; https://download.pytorch.org/whl/ and its cpu, cu126, cu128, cu129, cu130, cu132, rocm7.0, rocm7.1, rocm7.2, and rocm7.14 torch/ pages with their PEP 658 files; range reads of the 2.14.1+rocm7.2 wheel; https://repo.amd.com/rocm/whl-multi-arch/ (the rocm 7.14.1 sdist).
- Upstream source at tags: pytorch/pytorch v2.14.1; huggingface/transformers v5.18.0, v4.37.0, v4.36.2, v4.51.0, v4.50.3; huggingface/trl v1.14.1; huggingface/peft v0.21.2; huggingface/accelerate v1.15.0; astral-sh/uv 0.12.17 (docs/guides/integration/pytorch.md; docs/concepts/indexes.md, read 2026-10-06); llvm/llvm-project llvmorg-21.1.0 (offload-arch, ELF.h); ROCm/rocm-systems rocm-7.2.1 (libhsakmt openclose.c; fmm.c and topology.c, read 2026-10-06); the rocm 7.14.1 sdist (setup.py, src/rocm_sdk/_dist_info.py); vllm-project/vllm v0.31.0; sgl-project/sglang v0.5.21; ggml-org/llama.cpp v0.6.0.
- Hugging Face Hub, read 2026-10-05: the API and files at the revisions in Results 6 for Qwen/Qwen2.5-Coder-0.5B-Instruct, Qwen/Qwen2.5-0.5B-Instruct, Qwen/Qwen3-0.6B, and HuggingFaceTB/SmolLM2-360M-Instruct.
- The repository: lassi/ at 1aa7b0f and e805c6d (Results 7); projects/base.yaml, projects/lassi-demo/rngd-cpu.yaml and rngd-compile.yaml; plans/spikes/p17-frameworks/ (run.sh, frameworks.py, probe.hip, inspect_bundle.py); plans/p17-portable.md.
