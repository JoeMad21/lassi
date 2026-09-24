# P0.7 verification: pinned CUDA and NVHPC on alpha01

Date 2026-09-23 (host clock UTC-07:00). Device: none; alpha01 has no GPU or GPU driver, nothing ran on a device, and no built binary was executed. Installs were made by job 20260923-043820-toolchains-p07-dc38 (rc=0) from snapshot dce23ba of the install scripts. Runs 1 and 2 used `uv run tools/rx.py run` from the clean commit 4ef46ee, which holds the committed scripts. Run 3 used `uv run tools/rx.py exec`, which runs from the scratch root without a checkout and touches only the pinned installs. Every value below is [MEASURED] from the output shown. Lines cut from long output are marked `[... trimmed ...]`, except in runs 2a and 2b, which show only the compiler lines of their output and give the compiler invocations rather than the full rx command strings.

## Run 1: versions, checksum, sizes, root-filesystem check

rx id 20260923-045504-desktop-8r113ei-p0-core-8b48, rc=0, commit 4ef46ee, clean tree.

Command:

```
uv run tools/rx.py run --timeout 900 -- 'date -Is; git rev-parse HEAD; git status --short | wc -l; bash toolchains/cuda.sh; bash toolchains/nvhpc.sh; echo "--- pinned paths"; "$LASSI_TOOLCHAINS/cuda@12.6.3/bin/nvcc" --version; "$LASSI_TOOLCHAINS/nvhpc@24.11/Linux_x86_64/24.11/compilers/bin/nvc++" --version; echo "--- tarball"; ls -l "$LASSI_SCRATCH/downloads/nvhpc_2024_2411_Linux_x86_64_cuda_12.6.tar.gz"; sha256sum "$LASSI_SCRATCH/downloads/nvhpc_2024_2411_Linux_x86_64_cuda_12.6.tar.gz"; du -sh "$LASSI_TOOLCHAINS/cuda@12.6.3" "$LASSI_TOOLCHAINS/nvhpc@24.11"; echo "--- root tmp check"; find /tmp /var/tmp -xdev -user "$(id -un)" -newermt "2026-09-23 04:38:00" 2>/dev/null | head; echo "find done"'
```

Output:

```
2026-09-23T04:55:04-07:00
4ef46eeccce4ff4eee82da9edd2bb73ea9c15351
0
cuda: /mnt/nvme10/joseph_ufl/toolchains/cuda@12.6.3 already holds the pinned toolkit
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Tue_Oct_29_23:50:19_PDT_2024
Cuda compilation tools, release 12.6, V12.6.85
Build cuda_12.6.r12.6/compiler.35059454_0
nvhpc: /mnt/nvme10/joseph_ufl/toolchains/nvhpc@24.11 already holds the pinned SDK

nvc++ 24.11-0 64-bit target on x86-64 Linux -tp znver4
NVIDIA Compilers and Tools
Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
--- pinned paths
[... trimmed: the same nvcc and nvc++ version banners again, through the pinned paths ...]
--- tarball
-rw-rw-r-- 1 joseph_ufl joseph_ufl 6131381079 Sep 23 04:50 /mnt/nvme10/joseph_ufl/downloads/nvhpc_2024_2411_Linux_x86_64_cuda_12.6.tar.gz
39408ac062573936e0d41f34530ab065998ef4b85b6bcccb59287d71135a1920  /mnt/nvme10/joseph_ufl/downloads/nvhpc_2024_2411_Linux_x86_64_cuda_12.6.tar.gz
7.0G	/mnt/nvme10/joseph_ufl/toolchains/cuda@12.6.3
13G	/mnt/nvme10/joseph_ufl/toolchains/nvhpc@24.11
--- root tmp check
find done
```

- Both committed scripts found the pinned versions and skipped, so they are idempotent.
- The tarball size equals the server's content-length in rx 20260923-043558-exec-a651, whose Last-Modified was 2026-06-07.
- The CUDA runfile's md5 matched NVIDIA's published value during the install job.
- The root-filesystem check printed nothing.

## Run 2: nvc++ cross-compile for cc80 without a GPU

Source for run 2a, t.cpp:

```
#include <cstdio>
int main() {
  const int n = 1024; float x[n], y[n];
  for (int i = 0; i < n; i++) { x[i] = 1.0f; y[i] = 2.0f; }
  #pragma omp target teams distribute parallel for map(to: x[:n]) map(tofrom: y[:n])
  for (int i = 0; i < n; i++) y[i] += 3.0f * x[i];
  std::printf("%f\n", y[0]);
  return 0;
}
```

Run 2a: rx id 20260923-045538-desktop-8r113ei-p0-core-bde2, commit 4ef46ee, clean tree. Command, run in a directory under `$TMPDIR` that was removed afterwards:

```
LC_ALL=C "$LASSI_TOOLCHAINS/nvhpc@24.11/Linux_x86_64/24.11/compilers/bin/nvc++" -Wall -O3 -Minfo -mp=gpu -gpu=cc80 -o t t.cpp
```

Output:

```
nvc++-Error-A CUDA toolkit matching the current driver version (0) or a supported older version (11.8) was not installed with this HPC SDK.
nvc++ rc=1
```

Run 2b: rx id 20260923-045559-desktop-8r113ei-p0-core-778e, commit 4ef46ee, clean tree. The source is the same program with the `printf` removed and `return y[0] > 4.0f ? 0 : 1;` as the last statement of main. Commands:

```
LC_ALL=C NVHPC_CUDA_HOME="$LASSI_TOOLCHAINS/cuda@12.6.3" "$NVCPP" -Wall -O3 -Minfo -mp=gpu -gpu=cc80 -o ta t.cpp
LC_ALL=C "$NVCPP" -Wall -O3 -Minfo -mp=gpu -gpu=cc80,cuda12.6 -o tb t.cpp
"$LASSI_TOOLCHAINS/cuda@12.6.3/bin/cuobjdump" --list-elf ta
"$LASSI_TOOLCHAINS/cuda@12.6.3/bin/cuobjdump" --list-elf tb
```

`$NVCPP` is the pinned nvc++ path from run 2a.

Output:

```
[... trimmed: -Minfo lines for ta ...]
A rc=0
=== B: -gpu=cc80,cuda12.6
main:
      3, #omp target teams distribute parallel for
          3, Generating "nvkernel_main_F1L3_2" GPU kernel
          5, Loop parallelized across teams and threads(128), schedule(static)
[... trimmed ...]
B rc=0
--- ta
ELF file    1: pgcudafatjyJDkBrub01SA.sm_80.cubin
--- tb
ELF file    1: pgcudafatqDKDkW61w1XHW.sm_80.cubin
```

The adopted form is A: `NVHPC_CUDA_HOME` points at the pinned CUDA, and the bible's compile flags are unchanged. Form B changes the flags, so it was not adopted.

## Run 3: pinned nvcc cross-compile for sm_80 without a GPU

rx id 20260923-050157-exec-fa96 (`rx exec`, no checkout), rc=0. The program is the P0.6 kernel with a minimal host main, written to a directory under `$TMPDIR` that was removed afterwards. Output, including the source and the exact command:

```
2026-09-23T05:01:57-07:00
#include <cstdio>
__global__ void add(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}
int main() { std::printf("built\n"); return 0; }
--- /mnt/nvme10/joseph_ufl/toolchains/cuda@12.6.3/bin/nvcc -std=c++14 -Xcompiler -Wall -arch=sm_80 -O3 -o t t.cu
nvcc rc=0
ELF file    1: t.1.sm_80.cubin
ELF file    2: t.2.sm_80.cubin
list-elf rc=0
cleanup rc=0
```

## Conclusion

The pinned nvcc (cuda@12.6.3, run 3) and nvc++ (nvhpc@24.11, run 2b) both build sm_80 code on alpha01 without a GPU, with the bible's LASSI compile flags. nvc++ needs `NVHPC_CUDA_HOME` set to the pinned CUDA prefix, as recorded in toolchains/nvhpc.pin (CUDA_HOME_FROM) and plans/PHASE-NOTES.md for the runner.
