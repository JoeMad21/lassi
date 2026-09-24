# Spike P0.19: CUDA 12.6.3 from the redistributable archives

- Task: P0.19 (plans/p0-core.md). Bible: Toolchain Pins (OQ-010 decision of 2026-09-23), Agent Rules 7 and 10.
- Date: 2026-09-23, between 21:21 and 21:46 (alpha01 clock, UTC-07:00). Each rx id carries its own start time.
- Base: local commit 674bdd3 (`674bdd3cec5d2a3b293e71d520b3efaabbde0930`, branch p0-core). The probes before the install job used `uv run tools/rx.py exec`, which runs from the scratch root without a checkout, so no repository code took part in them. The install job and the checks after it used a dirty tree, which rx sent as a snapshot commit. Every result here is exploratory and not [MEASURED]; the reportable install runs from a clean commit.
- Host: alpha01, Ubuntu 22.04.5 LTS, GNU tar 1.34, bash 5.1.16, Python 3.10.12 (rx doctor, rx 20260923-214525-exec-757e). Gate environment: LASSI_SCRATCH = /mnt/nvme10/joseph_ufl, LASSI_TOOLCHAINS = /mnt/nvme10/joseph_ufl/toolchains, TMPDIR = /mnt/nvme10/joseph_ufl/tmp (rx 20260923-212151-exec-4f23).
- Device: none. alpha01 has no NVIDIA GPU or driver (bible Environment State); nothing opened a device and no built binary ran.
- Output conventions: outputs are verbatim except where `[... trimmed ...]` says otherwise.

## Question

Which of NVIDIA's CUDA 12.6.3 redistributable archives are enough to give, at the same prefix layout as the runfile install:

1. `bin/nvcc` reporting the pin's EXPECT_VERSION, "Cuda compilation tools, release 12.6, V12.6.85";
2. `-arch=sm_80` compiles of the P0.15 fixture scenarios and the HeCBench layout app with the LASSI flags;
3. a working NVHPC_CUDA_HOME for `nvc++ -gpu=cc80`?

Classification: factual.

## Disk before

rx 20260923-212151-exec-4f23:

```
2026-09-23T21:21:51-07:00
101G	/mnt/nvme10/joseph_ufl
104989528	/mnt/nvme10/joseph_ufl
Filesystem          1B-blocks          Used    Available Use% Mounted on
/dev/nvme23n1p1 3779301580800 3320575512576 420301746176  89% /mnt/nvme10
[... trimmed: an empty downloads directory, the toolchains listing ...]
7.0G	/mnt/nvme10/joseph_ufl/toolchains/cuda@12.6.3
13G	/mnt/nvme10/joseph_ufl/toolchains/nvhpc@24.11
```

The runfile tree at cuda@12.6.3 is 7.0G. The owner's cap is 120G on the scratch root; the probes below needed about 0.3G.

## Manifest

rx 20260923-212205-exec-19f2 fetched https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.6.3.json into $LASSI_SCRATCH/downloads and printed its sha256, its headers, and the linux-x86_64 entry of each component:

```
2026-09-23T21:22:05-07:00
-rw-rw-r-- 1 joseph_ufl joseph_ufl 49142 Sep 23 21:22 redistrib_12.6.3.json
9c598598457a6463eb92889080c16b2b9dc04150e501b8bfc1536d403ba70aaf  redistrib_12.6.3.json
content-length: 49142
etag: "9db5692edd83cb348dc16e6e1ae9b478:1732477827.531712"
last-modified: Wed, 20 Nov 2024 17:14:30 GMT
release_date 2024-11-20 release_label 12.6.3 product cuda
cuda_cccl 12.6.77 cuda_cccl/linux-x86_64/cuda_cccl-linux-x86_64-12.6.77-archive.tar.xz 9c3145ef01f73e50c0f5fcf923f0899c847f487c529817daa8f8b1a3ecf20925 934952
NOLINUX cuda_compat 12.6.36890662
cuda_cudart 12.6.77 cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-12.6.77-archive.tar.xz f74689258a60fd9c5bdfa7679458527a55e22442691ba678dcfaeffbf4391ef9 1126072
cuda_cuobjdump 12.6.77 cuda_cuobjdump/linux-x86_64/cuda_cuobjdump-linux-x86_64-12.6.77-archive.tar.xz 54d0edf14284249baf8e2db1c26c71d079ad7afd1f41779e22f239ff9e8bf60f 219068
[... trimmed: cuda_cupti through cuda_nsight ...]
cuda_nvcc 12.6.85 cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-12.6.85-archive.tar.xz 840deff234d9bef20d6856439c49881cb4f29423b214f9ecd2fa59b7ac323817 49996208
[... trimmed: the other 31 components, among them cuda_nvdisasm, cuda_nvrtc, libcublas, libnvjitlink, nsight_compute, nvidia_driver ...]
```

The fields per line are component, version, relative_path, sha256, and size in bytes. The fetch command:

```
uv run tools/rx.py exec -- 'date -Is; cd $LASSI_SCRATCH/downloads && curl -fsSL --retry 3 -o redistrib_12.6.3.json https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.6.3.json && ls -l redistrib_12.6.3.json && sha256sum redistrib_12.6.3.json && curl -fsSI https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.6.3.json | tr -d "\r" | grep -i -E "^(content-length|last-modified|etag)"; python3 -c "<prints the fields above for every component>"'
```

## Candidate archives

rx 20260923-212321-exec-84d0 downloaded cuda_nvcc, cuda_cudart, cuda_cccl, and cuda_cuobjdump and checked them with `sha256sum -c` against the manifest values:

```
2026-09-23T21:23:21-07:00
cuda_nvcc-linux-x86_64-12.6.85-archive.tar.xz: OK
cuda_cudart-linux-x86_64-12.6.77-archive.tar.xz: OK
cuda_cccl-linux-x86_64-12.6.77-archive.tar.xz: OK
cuda_cuobjdump-linux-x86_64-12.6.77-archive.tar.xz: OK
```

Each archive holds one top directory named after the file. Its top-level entries (rx 20260923-214525-exec-757e):

```
cuda_nvcc-linux-x86_64-12.6.85:  bin include lib LICENSE nvvm
cuda_cudart-linux-x86_64-12.6.77:  include lib LICENSE pkg-config
cuda_cccl-linux-x86_64-12.6.77:  include lib LICENSE
cuda_cuobjdump-linux-x86_64-12.6.77:  bin LICENSE
```

rx 20260923-212437-exec-1e14 extracted each archive into its own directory under $TMPDIR/p019-spike/parts and checked for paths that more than one component holds. It found none. nvcc.profile in cuda_nvcc is byte-identical to the runfile's, and the runfile prefix has two links:

```
--- overlaps (relative paths present in more than one component)
[... trimmed: the cudart lib listing: libcudadevrt.a, libcudart.so -> libcudart.so.12 -> libcudart.so.12.6.77, libcudart_static.a, libculibos.a, stubs/libcuda.so ...]
--- nvcc.profile

TOP              = $(_HERE_)/..

CICC_PATH        = $(TOP)/nvvm/bin
NVVMIR_LIBRARY_DIR = $(TOP)/nvvm/libdevice

LD_LIBRARY_PATH += $(TOP)/lib:
PATH            += $(CICC_PATH):$(_HERE_):

INCLUDES        +=  "-I$(TOP)/$(_TARGET_DIR_)/include" $(_SPACE_)

LIBRARIES        =+ $(_SPACE_) "-L$(TOP)/$(_TARGET_DIR_)/lib$(_TARGET_SIZE_)/stubs" "-L$(TOP)/$(_TARGET_DIR_)/lib$(_TARGET_SIZE_)"

CUDAFE_FLAGS    +=
PTXAS_FLAGS     +=
--- runfile nvcc.profile diff
same
lrwxrwxrwx  1 joseph_ufl joseph_ufl    28 Sep 23 04:42 include -> targets/x86_64-linux/include
lrwxrwxrwx  1 joseph_ufl joseph_ufl    24 Sep 23 04:42 lib64 -> targets/x86_64-linux/lib
```

## Spike prefix and compile checks

rx 20260923-212557-exec-c872 assembled a prefix in the runfile's layout at $TMPDIR/p019-spike/cuda:

- bin/ and nvvm/ at the top;
- include/ and lib/ under targets/x86_64-linux/;
- the links include and lib64.

The probe script:

```
for c in cuda_nvcc cuda_cudart cuda_cccl cuda_cuobjdump; do
  for e in $(ls -A "$s/parts/$c"); do
    case "$e" in
      bin|nvvm) cp -a "$s/parts/$c/$e" "$p/" ;;
      include|lib) cp -a "$s/parts/$c/$e" "$p/targets/x86_64-linux/" ;;
      LICENSE) ;;
      *) echo "UNKNOWN $c/$e" ;;
    esac
  done
done
ln -s targets/x86_64-linux/include "$p/include"
ln -s targets/x86_64-linux/lib "$p/lib64"
```

That run printed more than rx keeps, and the UNKNOWN line for cudart's pkg-config/ fell outside the captured output. The first install job found that entry (see below).

Its strace of nvc++ and nvcc showed which files under the prefix each one opens:

- nvcc, building layout-cuda: bin/nvcc, nvcc.profile, cudafe++, ptxas, fatbinary, nvlink, and crt/link.stub; nvvm/bin/cicc; headers under bin/../targets/x86_64-linux/include; and libcudadevrt.a and libcudart_static.a under targets/x86_64-linux/lib.
- nvc++, run 2b: bin/ptxas, bin/fatbinary, bin/nvlink, include/cuda.h, lib64, nvvm/lib64/libnvvm.so, and nvvm/libdevice/libdevice.10.bc.

The replay run, rx 20260923-212629-exec-290a, used the same prefix and printed the results in short form:

```
2026-09-23T21:26:29-07:00
--- spike prefix size
233316	/mnt/nvme10/joseph_ufl/tmp/p019-spike/cuda
--- a: nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Tue_Oct_29_23:50:19_PDT_2024
Cuda compilation tools, release 12.6, V12.6.85
Build cuda_12.6.r12.6/compiler.35059454_0
rc=0
--- b: run 3 program, sm_80 (rebuilt)
nvcc rc=0
ELF file    1: t.1.sm_80.cubin
ELF file    2: t.2.sm_80.cubin
list-elf rc=0
--- c: HeCBench layout-cuda, both installs
layout new rc=0 stderr bytes 0 sha256 e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
layout old rc=0 stderr bytes 0 sha256 e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
ELF file    1: main.1.sm_80.cubin
ELF file    2: main.2.sm_80.cubin
list-elf rc=0
layout stderr identical
--- d: nvc++ run 2b with NVHPC_CUDA_HOME at the spike prefix
[... trimmed: the -Minfo report, as in run 2b of plans/spikes/p0-toolchains-verify.md ...]
A rc=0
ELF file    1: pgcudafatfxqCopc4Iihym.sm_80.cubin
list-elf rc=0
[... trimmed: the strace summary above ...]
--- d3: missing paths under the spike prefix nvc++ looked for (not library search paths)
[... trimmed: nvvm/lib64/x86_64, nvvm-prev/, libdevice.compute_*.10.bc ...]
"/version.json"
"/version.txt"
```

The commands are the P0.7 checks (plans/spikes/p0-toolchains-verify.md, runs 2b and 3) with the bible's LASSI flags:

```
nvcc -std=c++14 -Xcompiler -Wall -arch=sm_80 -O3 -o t t.cu
nvcc -std=c++14 -Xcompiler -Wall -arch=sm_80 -O3 -o main main.cu                 # layout-cuda, LC_ALL=C LANG=C
NVHPC_CUDA_HOME=<prefix> nvc++ -Wall -O3 -Minfo -mp=gpu -gpu=cc80 -o ta t.cpp    # LC_ALL=C
cuobjdump --list-elf <binary>
```

nvc++ looks for version.json and version.txt under NVHPC_CUDA_HOME, finds neither in the archive install, reads include/cuda.h, and builds. The runfile tree has a version.json, which no archive carries.

## Fixture replay

rx 20260923-212704-exec-ab12 replayed every scenario in the slot's committed tests/toolchains/fixtures/captures.json (capture rx 20260923-112105-desktop-8r113ei-p0-core-ba1a, commit ebe6b07). Each scenario ran its recorded argv in a fresh copy of its sources, once with the spike prefix ("new") and once with the runfile prefix ("old"). The environment was PATH, LANG=C, LC_ALL=C, a private TMPDIR under the workdir, and NVHPC_CUDA_HOME for nvc++. The replay compares the stderr sha256 with the recorded one:

```
nvcc_clean rc=0 old:rc=0,same new:rc=0,same
nvcc_fatal rc=1 old:rc=1,same new:rc=1,same
nvcc_host_gcc_warning rc=0 old:rc=0,same new:rc=0,same
nvcc_linker_error rc=1 old:rc=1,DIFF(301 bytes) new:rc=1,DIFF(301 bytes)
nvcc_ptxas_error rc=255 old:rc=255,same new:rc=255,same
nvcc_undefined_identifier rc=2 old:rc=2,same new:rc=2,same
nvcc_warning_177 rc=0 old:rc=0,same new:rc=0,same
nvcpp_backend_error rc=2 old:rc=2,same new:rc=2,same
nvcpp_edg_error rc=2 old:rc=2,same new:rc=2,same
nvcpp_edg_warning rc=0 old:rc=0,same new:rc=0,same
nvcpp_fatal_abort rc=2 old:rc=2,same new:rc=2,same
nvcpp_linker_error rc=2 old:rc=2,DIFF(796 bytes) new:rc=2,DIFF(796 bytes)
nvcpp_minfo_clean rc=0 old:rc=0,same new:rc=0,same
nvcpp_missing_include rc=2 old:rc=2,same new:rc=2,same
```

- The 12 byte-stable scenarios match the recorded sha256 with both prefixes.
- The two linker errors differ with both prefixes, since their stderr names the workdir. Between new and old they differ only in that workdir path.
- The replay is not the capture tool, and it did not use the compile sandbox. The reportable recapture is the tool's, from a clean commit.

## Why CCCL is needed

cuda_fp16.h ships in cuda_cudart and includes nv/target, which cuda_cccl ships. rx 20260923-212738-exec-59b0 compiled a one-line use of `__half` without cccl and then with it:

```
== prefix /mnt/nvme10/joseph_ufl/tmp/p019-spike/nocccl
In file included from h.cu:1:
/mnt/nvme10/joseph_ufl/tmp/p019-spike/nocccl/bin/../targets/x86_64-linux/include/cuda_fp16.h:4410:10: fatal error: nv/target: No such file or directory
 4410 | #include <nv/target>
      |          ^~~~~~~~~~~
rc=1
== prefix /mnt/nvme10/joseph_ufl/tmp/p019-spike/cuda
rc=0
```

## Same bytes as the runfile install

rx 20260923-212753-exec-288c compared every file of the spike prefix with the file at the same path in the runfile tree, using `cmp`:

```
files=1700 identical=1700 differ=0 missing=0
symlinks in the new tree: 6
93032	bin
57440	nvvm
82840	targets
```

All six links (include, lib64, and four library links) name the same targets as in the runfile tree.

## Decision (toolchains/cuda.pin)

- Components: cuda_nvcc 12.6.85, cuda_cudart 12.6.77, cuda_cccl 12.6.77, and cuda_cuobjdump 12.6.77.
  - cuda_nvcc holds nvcc, ptxas, nvlink, fatbinary, cudafe++, bin2c, crt/, cicc, libnvvm, and libdevice.
  - cuda_cudart holds the runtime headers, including cuda.h, and libcudart and libcudadevrt.
  - cuda_cccl provides nv/target for cuda_fp16.h.
  - cuda_cuobjdump is needed only for the install's own sm_80 checks and the P0.7 checks, which list the cubins.
- Layout: the runfile's.
  - bin/ and nvvm/ go at the prefix.
  - include/ and lib/ go under targets/x86_64-linux/, with the links include and lib64.
  - Each LICENSE goes under licenses/<component>/.
  - cuda_cudart's pkg-config/ is left out (LAYOUT_SKIP). Its .pc files name a system CUDA root, which the runfile rewrote to the prefix, and no LASSI compile uses pkg-config.
- The pin records:
  - the manifest URL and sha256;
  - each component's version, size, sha256, and relative path;
  - the layout;
  - a planning bound of 250000 KiB for the prefix (233316 KiB measured here, rounded up).
- PREFIX_NAME and EXPECT_VERSION are unchanged.

## Exploratory install job

Before the job, rx 20260923-213947-exec-3ece removed this spike's own files: the five downloads it had made and $TMPDIR/p019-spike. That way the job downloads for real. It then measured the scratch root:

```
101G	/mnt/nvme10/joseph_ufl
104989712	/mnt/nvme10/joseph_ufl
```

rx 20260923-214013-exec-acf3 also removed three 2-byte nvacc*.bc files (21:26:01 and 21:26:33) that the spike's nvc++ builds had left in $TMPDIR (rx 20260923-214002-exec-22e1 listed them). The install script therefore gives its check compiles a private TMPDIR.

`uv run tools/rx.py doctor` reported stop false, 420.3 GB free on the scratch disk, and no running jobs.

First job, rx 20260923-214050-cuda-redist-b5c8: `uv run tools/rx.py job start --big --name cuda-redist -- 'bash toolchains/cuda.sh'`, from a dirty tree (snapshot 333c7015402f of 674bdd3). It failed closed with rc=1 on cudart's pkg-config/, which the layout did not place yet:

```
cuda: space check passed: 104989944 KiB used under /mnt/nvme10/joseph_ufl, 301053 KiB planned, cap 125829120 KiB
cuda: downloaded /mnt/nvme10/joseph_ufl/downloads/redistrib_12.6.3.json (sha256 9c598598457a6463eb92889080c16b2b9dc04150e501b8bfc1536d403ba70aaf)
cuda: downloaded /mnt/nvme10/joseph_ufl/downloads/cuda_nvcc-linux-x86_64-12.6.85-archive.tar.xz (sha256 840deff234d9bef20d6856439c49881cb4f29423b214f9ecd2fa59b7ac323817)
cuda: placed cuda_nvcc 12.6.85
cuda: downloaded /mnt/nvme10/joseph_ufl/downloads/cuda_cudart-linux-x86_64-12.6.77-archive.tar.xz (sha256 f74689258a60fd9c5bdfa7679458527a55e22442691ba678dcfaeffbf4391ef9)
cuda: /mnt/nvme10/joseph_ufl/downloads/cuda_cudart-linux-x86_64-12.6.77-archive.tar.xz holds pkg-config, which the layout in cuda.pin does not place
```

The script removed its own staging prefix, and the runfile tree stayed in place. The pin then gained LAYOUT_SKIP="pkg-config".

Second job, rx 20260923-214305-cuda-redist-7b27, the same command from a dirty tree (snapshot 9769020ee332 of 674bdd3), ran from 21:43:05 to 21:43:25 with rc=0:

```
cuda: space check passed: 105040008 KiB used under /mnt/nvme10/joseph_ufl, 301053 KiB planned, cap 125829120 KiB
cuda: reusing /mnt/nvme10/joseph_ufl/downloads/redistrib_12.6.3.json (sha256 9c598598457a6463eb92889080c16b2b9dc04150e501b8bfc1536d403ba70aaf)
cuda: reusing /mnt/nvme10/joseph_ufl/downloads/cuda_nvcc-linux-x86_64-12.6.85-archive.tar.xz (sha256 840deff234d9bef20d6856439c49881cb4f29423b214f9ecd2fa59b7ac323817)
cuda: placed cuda_nvcc 12.6.85
cuda: reusing /mnt/nvme10/joseph_ufl/downloads/cuda_cudart-linux-x86_64-12.6.77-archive.tar.xz (sha256 f74689258a60fd9c5bdfa7679458527a55e22442691ba678dcfaeffbf4391ef9)
cuda: leaving cuda_cudart's pkg-config out of the install (LAYOUT_SKIP in cuda.pin)
cuda: placed cuda_cudart 12.6.77
cuda: downloaded /mnt/nvme10/joseph_ufl/downloads/cuda_cccl-linux-x86_64-12.6.77-archive.tar.xz (sha256 9c3145ef01f73e50c0f5fcf923f0899c847f487c529817daa8f8b1a3ecf20925)
cuda: placed cuda_cccl 12.6.77
cuda: downloaded /mnt/nvme10/joseph_ufl/downloads/cuda_cuobjdump-linux-x86_64-12.6.77-archive.tar.xz (sha256 54d0edf14284249baf8e2db1c26c71d079ad7afd1f41779e22f239ff9e8bf60f)
cuda: placed cuda_cuobjdump 12.6.77
ELF file    1: t.1.sm_80.cubin
ELF file    2: t.2.sm_80.cubin
cuda: nvcc built an sm_80 cubin
[... trimmed: the -Minfo report ...]
ELF file    1: pgcudafatWVMdcqjTYvAhS.sm_80.cubin
cuda: nvc++ built an sm_80 cubin with NVHPC_CUDA_HOME at /mnt/nvme10/joseph_ufl/toolchains/cuda@12.6.3.staging
cuda: moved the previous install aside to /mnt/nvme10/joseph_ufl/toolchains/cuda@12.6.3.runfile-20260923-214325
cuda: installed cuda@12.6.3 from the redistributable archives
cuda: the previous install stays at /mnt/nvme10/joseph_ufl/toolchains/cuda@12.6.3.runfile-20260923-214325 until the owner decides to remove it
233596	/mnt/nvme10/joseph_ufl/toolchains/cuda@12.6.3
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Tue_Oct_29_23:50:19_PDT_2024
Cuda compilation tools, release 12.6, V12.6.85
Build cuda_12.6.r12.6/compiler.35059454_0
```

The script ran its own root-filesystem check before the swap, a find over /tmp and /var/tmp for files changed after its start marker, and it passed. That version discarded find's errors, so a find that could not look would also have passed; the check was then made to read them (see "Script change after review").

## Checks after the job

rx 20260923-214437-desktop-8r113ei-p0-core-0bb1 (`uv run tools/rx.py run`, snapshot 686fd1845f19 of 674bdd3) ran:

- the install script again;
- `nvcc --version` through the pinned path;
- the P0.7 run 3 and run 2b checks against the installed prefix, plus layout-cuda;
- the fixture replay against the snapshot's captures.json;
- the disk use;
- the find over /tmp and /var/tmp.

```
2026-09-23T21:44:37-07:00
--- 1: rerun of the install script (idempotency)
cuda: /mnt/nvme10/joseph_ufl/toolchains/cuda@12.6.3 already holds the pinned toolkit
[... trimmed: the nvcc banner ...]
cuda.sh rc=0
--- 2: nvcc --version through the pinned path
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Tue_Oct_29_23:50:19_PDT_2024
Cuda compilation tools, release 12.6, V12.6.85
Build cuda_12.6.r12.6/compiler.35059454_0
rc=0
--- 3: layout
[... trimmed: ls -la of the prefix: bin, include -> targets/x86_64-linux/include, lib64 -> targets/x86_64-linux/lib, licenses, nvvm, redist-components.txt, targets ...]
licenses: cuda_cccl cuda_cudart cuda_cuobjdump cuda_nvcc
bin: bin2c crt cudafe++ cuobjdump fatbinary nvcc __nvcc_device_query nvcc.profile nvlink ptxas
nvvm: bin include lib64 libdevice
targets/x86_64-linux: include lib
[... trimmed: redist-components.txt, the manifest, archive, and layout lines of toolchains/cuda.pin ...]
233596	/mnt/nvme10/joseph_ufl/toolchains/cuda@12.6.3
7300232	/mnt/nvme10/joseph_ufl/toolchains/cuda@12.6.3.runfile-20260923-214325
--- 4a: P0.7 run 3 check, nvcc sm_80
nvcc rc=0
ELF file    1: t.1.sm_80.cubin
ELF file    2: t.2.sm_80.cubin
list-elf rc=0
--- 4b: P0.7 run 2b check, nvc++ cc80 with NVHPC_CUDA_HOME
[... trimmed: the -Minfo report ...]
A rc=0
ELF file    1: pgcudafattvMsc5qKwoZ8r.sm_80.cubin
list-elf rc=0
--- 4c: HeCBench layout-cuda with the LASSI flags
layout rc=0 stderr bytes 0
ELF file    1: main.1.sm_80.cubin
ELF file    2: main.2.sm_80.cubin
list-elf rc=0
--- 5: replay of the captured fixture scenarios against captures.json
captures.json commit 674bdd3cec5d2a3b293e71d520b3efaabbde0930 rx 20260923-211958-desktop-8r113ei-p0-core-d221
[... trimmed: 12 lines "rc N recorded N stderr same" ...]
nvcc_linker_error rc 1 recorded 1 stderr DIFF
nvcc_ptxas_inline_asm rc 255 recorded 255 stderr DIFF
nvcpp_linker_error rc 2 recorded 2 stderr DIFF
nvcpp_nvlink_error rc 2 recorded 2 stderr DIFF
identical 12 of 16
--- 6: scratch tmp leftovers from the install
none
cuda@12.6.3
cuda@12.6.3.runfile-20260923-214325
nvhpc@24.11
--- 7: disk
101G	/mnt/nvme10/joseph_ufl
105274768	/mnt/nvme10/joseph_ufl
--- 8: root filesystem check since the job start
find since the job start done
find since the first spike probe done
```

- The two finds, over /tmp and /var/tmp with `-xdev -user joseph_ufl`, used `-newermt "2026-09-23 21:43:05"` (the job start) and `-newermt "2026-09-23 21:21:00"` (before the first probe). Neither printed anything.
- The replay used the snapshot's captures.json, which holds the 16 scenarios of the uncommitted P0.17 recapture. The 12 byte-stable scenarios matched the recorded sha256, and all 16 kept their recorded exit status. The four that differ name a temporary or work path in their stderr.

## Disk after

- The scratch root went from 104989528 KiB to 105274768 KiB, 101G in `du -sh` both times. The difference is 285240 KiB.
- The new prefix is 233596 KiB, and the downloads are 51112 KiB (rx 20260923-214525-exec-757e).
- The runfile tree, 7300232 KiB, stays at toolchains/cuda@12.6.3.runfile-20260923-214325. Removing it is an owner decision.

## Script change after review

Review of the change found two gaps in toolchains/cuda.sh, and both were fixed:

- The root-filesystem check sent find's errors to /dev/null and ignored its exit status, so a find that could not look passed as a clean one. The check is now the function check_root_fs. It refuses a start directory that is not there, keeps find's stderr (with LC_ALL=C), ignores and counts "Permission denied" lines, and fails on any other error line or on a nonzero exit with no permission-denied line. It prints the number of unreadable paths it skipped.
- The swap was tested only by the order of its text. It is now the function swap_in, and tests/toolchains/test_pins.py runs it in bash on local directories. The tests cover a first install, a runfile tree and a redist tree moved aside, a failed rename that puts the old tree back, a failed put-back that leaves the old tree aside, and an aside name that is already taken.

rx 20260923-220331-desktop-8r113ei-p0-core-b2aa (`uv run tools/rx.py run`, snapshot 91d19f960c50 of 674bdd3, dirty tree, exploratory) ran the changed script once, from 22:03:31 to 22:03:52. It installed into a scratch toolchains root, $LASSI_SCRATCH/tmp/p019-fix/toolchains, so the swap and the new check ran without touching the installed prefix. That root held a stand-in "runfile tree" at cuda@12.6.3, which was a directory with one text file and no component record, and a link to the pinned nvhpc@24.11 for the nvc++ check. Output, with the -Minfo report and the repeated nvcc banners trimmed:

```
2026-09-23T22:03:31-07:00
slot commit 91d19f960c5089311f14584d8a3e58d0ed8b4f9f rx snapshot of 674bdd3cec5d
--- 1: install into a scratch toolchains root over a stand-in runfile tree
cuda: space check passed: 105274968 KiB used under /mnt/nvme10/joseph_ufl, 301053 KiB planned, cap 125829120 KiB
[... trimmed: the manifest and the four archives reused with their sha256, pkg-config left out, each component placed ...]
ELF file    1: t.1.sm_80.cubin
ELF file    2: t.2.sm_80.cubin
cuda: nvcc built an sm_80 cubin
[... trimmed: the -Minfo report ...]
ELF file    1: pgcudafatSLOngeZWfPysc.sm_80.cubin
cuda: nvc++ built an sm_80 cubin with NVHPC_CUDA_HOME at /mnt/nvme10/joseph_ufl/tmp/p019-fix/toolchains/cuda@12.6.3.staging
cuda: root-filesystem check passed: nothing changed under /tmp /var/tmp since the start (find rc=1, 85 unreadable paths skipped)
cuda: moved the previous install aside to /mnt/nvme10/joseph_ufl/tmp/p019-fix/toolchains/cuda@12.6.3.runfile-20260923-220350
cuda: installed cuda@12.6.3 from the redistributable archives
cuda: the previous install stays at /mnt/nvme10/joseph_ufl/tmp/p019-fix/toolchains/cuda@12.6.3.runfile-20260923-220350 until the owner decides to remove it
233596	/mnt/nvme10/joseph_ufl/tmp/p019-fix/toolchains/cuda@12.6.3
[... trimmed: the nvcc banner, "Cuda compilation tools, release 12.6, V12.6.85" ...]
cuda.sh rc=0
--- 2: result
[... trimmed: ls -la: cuda@12.6.3, cuda@12.6.3.runfile-20260923-220350, nvhpc@24.11 -> /mnt/nvme10/joseph_ufl/toolchains/nvhpc@24.11 ...]
stand-in for a runfile tree
[... trimmed: the new prefix's entries: bin include lib64 licenses nvvm redist-components.txt targets ...]
--- 3: rerun over the new prefix (idempotency)
cuda: /mnt/nvme10/joseph_ufl/tmp/p019-fix/toolchains/cuda@12.6.3 already holds the pinned toolkit
rc=0
--- 4: rerun against the installed prefix
cuda: /mnt/nvme10/joseph_ufl/toolchains/cuda@12.6.3 already holds the pinned toolkit
rc=0
--- 5: independent find since 2026-09-23 22:03:31 over /tmp and /var/tmp, errors other than permission denied kept
0
85
--- 6: remove this run scratch root
[... trimmed: the listing of $LASSI_SCRATCH/tmp, which no longer holds p019-fix ...]
105274956	/mnt/nvme10/joseph_ufl
2026-09-23T22:03:52-07:00
```

- In section 5, the first count is the number of find output lines that are not permission-denied errors, which is none: no changed file and no other error. The second count is the number of permission-denied lines, which matches the 85 the script reported.
- The stand-in tree was kept whole under its runfile- name. The scratch root went back to 105274956 KiB after the run removed its own p019-fix directory.
- rx 20260923-220419-exec-644e listed the entries of $LASSI_SCRATCH/tmp changed since 22:03:31. Only the directory itself had changed, from creating and removing p019-fix and the start marker. The nvacc*.bc, nvc++*, and pgcuda* files in that directory date from 17:35 or earlier and do not come from this task.
- The snapshot commits of the earlier runs come from their gate records (rx 20260923-215931-exec-db89): 333c7015402f for the first job, 9769020ee332 for the second, and 686fd1845f19 for the checks after the job. All three are snapshots of 674bdd3.

## Findings (exploratory)

- The four archives give the pinned nvcc at the runfile's prefix layout, with its EXPECT_VERSION.
- Their 1700 files are byte-identical to the runfile's files at the same paths.
- nvcc sm_80 builds pass for the P0.7 program and layout-cuda.
- nvc++ -gpu=cc80 builds pass with NVHPC_CUDA_HOME at the new prefix.
- The 12 byte-stable fixture scenarios give byte-identical stderr in a replay.
- The install writes nothing to /tmp or /var/tmp.

## Not covered here

- The reportable install run and the P0.15 fixture recapture with tools/capture_toolchain_fixtures.py, both from a clean commit.
- Removing the runfile tree set aside at toolchains/cuda@12.6.3.runfile-20260923-214325 (an owner queue item).
- Components that later phases may need, for example cuBLAS, cuRAND, or NVRTC for HeCBench apps that link them. Each addition is a pin change and a Decision Log entry.
