# Spike P0.6: nvcc and nvc++ on alpha01

- Task: P0.6 in `plans/p0-core.md`.
- Date: 2026-09-23. Host timestamps come from `date -Is` on alpha01 (UTC-07:00). The rx run ids embed the same host clock.
- Local checkout: `22f22154d25e45ab42d68f734a1f79d67497a374` on branch `p0-core`. The working tree was dirty only in `plans/STATUS.md` (task-state edit). Every remote command below used `rx exec`, which runs from the scratch root without a checkout, so no repository code took part in any result.
- Host: alpha01 (I/ONX), reached only through `uv run tools/rx.py`.

## Question

Which of nvcc and nvc++ exist on alpha01, at which versions and paths, and do they cross-compile for sm_80 without a GPU?

Why it matters:

- The bible's Execution Backends row for executor `none` says "nvcc and nvc++ cross-compile for sm_80 without a GPU | Available". This spike checks that Status.
- P0.7 either pins usable existing installs or installs CUDA and NVHPC under `$LASSI_TOOLCHAINS`.
- P0.15 replaces the hand-written toolchain stderr fixtures with real captures. This spike records the first real nvcc stderr as evidence only; no test was edited.

## Classification

Factual. It can be answered by running read-only commands and a compile on alpha01.

## Method

- Every remote command went through `uv run tools/rx.py doctor` or `uv run tools/rx.py exec`. No raw ssh, scp, rsync, or sftp was used.
- No device command was run: no devcheck, no `nvidia-smi`, and no `/dev` probing. Device classes are disabled in the gate, and this spike needs none.
- Temporary files lived only in `mktemp -d "$TMPDIR/p0-nvcc.XXXXXX"`, where `TMPDIR=/mnt/nvme10/joseph_ufl/tmp` is set by the gate. Each probe removed its directory and checked the removal.
- Built binaries were never run.
- Output conventions:
  - Outputs are verbatim except where a `[... trimmed ...]` marker says otherwise.
  - Leading tab characters in the `ldd` and `cuobjdump` output are rendered as spaces.
  - The only non-ASCII bytes in any output were GCC's UTF-8 quotes under the gate's default `LANG=en_US.UTF-8`. They appear only inside `cat -A` output, which escapes them: `M-bM-^@M-^X` is the byte sequence E2 80 98 (U+2018, left single quote), and `M-bM-^@M-^Y` is E2 80 99 (U+2019, right single quote). In `cat -A` output, `$` marks each end of line.

## Commands and Outputs

### 1. rx doctor

Time: 2026-09-23T11:13:06Z (04:13:06-07:00), from the local clock immediately before the call. `rx doctor` prints no host date; the host date in step 2 was taken 10 seconds later.

Command:

```
uv run tools/rx.py doctor
```

Output (full), exit status 0:

```
[rx] settings: {"host": "ionx", "scratch": "/mnt/nvme10/joseph_ufl", "remote_python": "python3", "machine": "desktop-8r113ei", "transport": "ssh"}
[rx] ssh ionx: PASS
{
  "gate_version": 1,
  "scratch": "/mnt/nvme10/joseph_ufl",
  "stop": false,
  "scratch_free_gb": 550.6,
  "root_free_gb": 321.3,
  "host": {
    "hostname": "alpha01",
    "kernel": "6.6.29+main+3.0.0r1-amd64-gio-epilmore-dev+",
    "python": "3.10.12",
    "nproc": 256,
    "os": "Ubuntu 22.04.5 LTS",
    "mem_gb": 3169.6,
    "mem_avail_gb": 2986.4,
    "load": [
      11.0546875,
      7.908203125,
      9.45703125
    ]
  },
  "bare_repo": true,
  "git": "git version 2.34.1",
  "uv": "/mnt/nvme10/joseph_ufl/bin/uv",
  "config": {
    "min_free_gb": 5,
    "min_free_gb_big": 60,
    "max_jobs": 3,
    "max_big_jobs": 1,
    "max_build_jobs": 32
  },
  "devices_enabled": {
    "rngd": false,
    "tt_silicon": false,
    "rocm_gpu": false,
    "nvidia_gpu": false
  },
  "running_jobs": [],
  "slots": [],
  "ok": true
}
[rx] Ready
```

### 2. Acceptance probe

The host date was taken by a separate exec just before the probe, so the probe itself stays exactly as the acceptance criterion names it.

```
uv run tools/rx.py exec -- 'date -Is'
```

```
2026-09-23T04:13:16-07:00

[rx] id=20260923-041316-exec-8d2d rc=0 state=done
```

Command (verbatim from the acceptance criterion):

```
uv run tools/rx.py exec -- 'command -v nvcc nvc++; ls /usr/local /opt; df -h /mnt/nvme10'
```

Output (full), rx id `20260923-041318-exec-de88`:

```
/opt:
ama_portal
amd
amdgpu
amd-libdrm
ansible-runtime
ansible-runtime-constraints-838389db5a48e9bc6450e1ea51c4b36dd3d3f88f.txt
ansible-runtime-constraints-b76491a016dd12987f51a01b75625b405063f669.txt
axelera
cache
cni
containerd
deployed-versions
discovery-client
eq
gigaio
gigaio-devwork
grpc
ionx
ionx-sdk
lxdware.crt
openmpi-v5.0.7-ulfm
openstack-ansible
rocm
rocm-7.2.0
rust
stack
tenstorrent

/usr/local:
bin
cargo
doc
etc
games
include
lib
man
rustup
sbin
share
src
Filesystem       Size  Used Avail Use% Mounted on
/dev/nvme23n1p1  3.5T  3.0T  513G  86% /mnt/nvme10

[rx] id=20260923-041318-exec-de88 rc=0 state=done
```

Notes:

- `command -v nvcc nvc++` printed nothing, so neither compiler is on the gate's PATH. The overall rc=0 comes from the last command (`df`). Step 3a records `command -v` exiting 1 on its own.
- `ls` sorts its operands, so `/opt` is listed before `/usr/local`. Neither contains `cuda*` or `nvidia`.

### 3. Search for installs not on PATH

#### 3a. Environment, standard locations, modules, alternatives

```
uv run tools/rx.py exec -- 'date -Is; command -v nvcc nvc++; echo "command-v rc=$?"; echo "PATH=$PATH"; echo "LASSI_TOOLCHAINS=$LASSI_TOOLCHAINS"; echo "TMPDIR=$TMPDIR"; echo "LASSI_SCRATCH=$LASSI_SCRATCH"; echo "--- cuda/hpc_sdk dirs"; ls -d /usr/local/cuda* /opt/nvidia /opt/nvidia/hpc_sdk/*/* /usr/lib/cuda /usr/lib/nvidia-cuda-toolkit 2>/dev/null; echo "ls rc=$?"; echo "--- LASSI_TOOLCHAINS"; ls -la "$LASSI_TOOLCHAINS" 2>&1; echo "--- module avail"; (module avail 2>&1 || echo "module: not available") | head -50; echo "--- alternatives"; ls /etc/alternatives | grep -i -E "cuda|nv" ; echo "grep rc=$?"; echo "--- env CUDA vars"; env | grep -i -E "^(CUDA|NVHPC|NVCC)" ; echo "env rc=$?"'
```

```
2026-09-23T04:13:28-07:00
command-v rc=1
PATH=/mnt/nvme10/joseph_ufl/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games
LASSI_TOOLCHAINS=/mnt/nvme10/joseph_ufl/toolchains
TMPDIR=/mnt/nvme10/joseph_ufl/tmp
LASSI_SCRATCH=/mnt/nvme10/joseph_ufl
--- cuda/hpc_sdk dirs
ls rc=2
--- LASSI_TOOLCHAINS
total 8
drwxrwxr-x  2 joseph_ufl joseph_ufl 4096 Sep 22 18:46 .
drwxr-x--- 44 joseph_ufl joseph_ufl 4096 Sep 22 18:46 ..
--- module avail
bash: line 1: module: command not found
module: not available
--- alternatives
ALTER_CONVERSION.7.gz
CREATE_CONVERSION.7.gz
DROP_CONVERSION.7.gz
runvx
grep rc=0
--- env CUDA vars
env rc=1

[rx] id=20260923-041328-exec-abff rc=0 state=done
```

Notes:

- `/usr/local/cuda*`, `/opt/nvidia`, `/usr/lib/cuda`, and `/usr/lib/nvidia-cuda-toolkit` do not exist (`ls` rc=2).
- `$LASSI_TOOLCHAINS` is empty, and there are no environment modules and no CUDA or NVHPC environment variables.
- The alternatives matches come from the case-insensitive `nv` pattern (`conversion`, `runvx`); none is CUDA-related.

#### 3b. Root filesystem, /opt, packages, loader cache

```
uv run tools/rx.py exec --timeout 400 -- 'date -Is; echo "--- find / maxdepth 4 (nvcc, nvc++)"; timeout 180 find / -xdev -maxdepth 4 \( -name nvcc -o -name nvc++ \) -type f 2>/dev/null | head -20; echo "find-root rc=${PIPESTATUS[0]}"; echo "--- find /opt maxdepth 7"; timeout 120 find /opt -maxdepth 7 \( -name nvcc -o -name nvc++ -o -name "libcudart.so*" \) 2>/dev/null | head -20; echo "find-opt rc=${PIPESTATUS[0]}"; echo "--- dpkg"; dpkg -l 2>/dev/null | grep -i -E "cuda|nvhpc|nvidia" | head -30; echo "dpkg-grep rc=${PIPESTATUS[1]}"; echo "--- ldconfig"; ldconfig -p 2>/dev/null | grep -i -E "cudart|libcuda\.so|nvrtc" | head; echo "ldconfig-grep rc=${PIPESTATUS[1]}"'
```

```
2026-09-23T04:13:41-07:00
--- find / maxdepth 4 (nvcc, nvc++)
find-root rc=1
--- find /opt maxdepth 7
find-opt rc=1
--- dpkg
ii  amdrocm-hipify7.12                                        7.12.0-1                                                         amd64        Hipify CUDA source
ii  hipify-clang                                              22.0.0.70200-43~22.04                                            amd64        Hipify CUDA source
dpkg-grep rc=0
--- ldconfig
ldconfig-grep rc=1

[rx] id=20260923-041341-exec-b16b rc=0 state=done
```

Notes:

- No matches. `find` exits 1 because some directories are unreadable; stderr was discarded.
- The only dpkg matches are AMD hipify tools.
- The loader cache has no `libcuda.so` (NVIDIA driver library), no `libcudart`, and no `libnvrtc`.
- `/opt` was searched to depth 7, which covers the default NVHPC layout `/opt/nvidia/hpc_sdk/Linux_x86_64/<rel>/compilers/bin/nvc++`.

#### 3c. Other filesystems and /mnt/nvme10

```
uv run tools/rx.py exec --timeout 500 -- 'date -Is; echo "--- mounts (non-virtual)"; df -h -x tmpfs -x devtmpfs -x overlay -x squashfs 2>/dev/null | head -40; echo "--- ls /mnt"; ls /mnt | head -40; echo "--- find /mnt/nvme10 maxdepth 5"; timeout 240 find /mnt/nvme10 -maxdepth 5 \( -name nvcc -o -name nvc++ -o -name "cuda-1*" -o -name hpc_sdk \) 2>/dev/null | head -30; echo "find-nvme10 rc=${PIPESTATUS[0]}"'
```

```
2026-09-23T04:13:52-07:00
--- mounts (non-virtual)
Filesystem                         Size  Used Avail Use% Mounted on
efivarfs                           256K  122K  130K  49% /sys/firmware/efi/efivars
/dev/mapper/ubuntu--vg-ubuntu--lv  1.7T  1.3T  300G  82% /
/dev/loop21                         11G  572M  9.7G   6% /var/lib/machines
/dev/loop22                        1.0T  7.2G 1017G   1% /var/lib/nova/instances
/dev/loop19                        1.0T  7.2G 1017G   1% /srv/swift2.img
/dev/loop20                        1.0T  7.2G 1017G   1% /srv/swift3.img
/dev/loop18                        1.0T  7.2G 1017G   1% /srv/swift1.img
/dev/nvme2n1p2                     4.9G  435M  4.3G  10% /boot
/dev/nvme19n1p1                    3.5T  3.2T  113G  97% /mnt/nvme0
/dev/nvme1n1p1                     3.5T  3.2T   71G  98% /mnt/nvme1
/dev/nvme23n1p1                    3.5T  3.0T  513G  86% /mnt/nvme10
/dev/nvme3n1p1                     7.0T  6.4T  658G  91% /mnt/llama
/dev/nvme0n1p1                     3.5T  3.1T  172G  95% /mnt/nvme2
/dev/nvme21n1p1                    3.5T  311G  3.2T   9% /mnt/nvme7
/dev/nvme20n1p1                    3.5T  2.9T  406G  88% /mnt/nvme3
/dev/nvme2n1p1                     1.1G  6.1M  1.1G   1% /boot/efi
--- ls /mnt
april-alpha_drew
data
IONXBackup
kv_caching
llama
lxd-storage
moe_rocm
nmve1
nvme0
nvme1
nvme10
nvme2
nvme3
nvme4
nvme5
nvme6
nvme7
nvme8
nvme9
nvmeof
nvmeof-test
sharding
storage_array01
--- find /mnt/nvme10 maxdepth 5
/mnt/nvme10/joseph_ufl/cuda-12.6.3
/mnt/nvme10/joseph_ufl/cuda-12.6.3/pkgconfig/cuda-12.6.pc
/mnt/nvme10/joseph_ufl/cuda-12.6.3/bin/nvcc
find-nvme10 rc=1

[rx] id=20260923-041352-exec-af03 rc=0 state=done
```

#### 3d. Contents of /mnt/nvme10/joseph_ufl/cuda-12.6.3

```
uv run tools/rx.py exec -- 'date -Is; C=/mnt/nvme10/joseph_ufl/cuda-12.6.3; ls -la $C; echo "--- bin"; ls -la $C/bin; echo "--- nvvm"; ls -la $C/nvvm $C/nvvm/bin 2>&1 | head -20; echo "--- lib64 / targets"; ls $C/lib64 2>&1 | head -30; ls $C/targets 2>&1; echo "--- file nvcc"; file $C/bin/nvcc; echo "--- version.json"; head -c 1500 $C/version.json 2>&1; echo; echo "--- du"; timeout 60 du -sh $C 2>&1'
```

The output was 185 lines. rx showed the last 120, and the first lines were read back with `uv run tools/rx.py job tail -n 250 20260923-041407-exec-df26`. The output is trimmed here as marked.

```
2026-09-23T04:14:07-07:00
total 148
drwxrwxr-x 18 joseph_ufl joseph_ufl  4096 Jul  8 12:25 .
drwxr-x--- 44 joseph_ufl joseph_ufl  4096 Sep 22 18:46 ..
drwxrwxr-x  3 joseph_ufl joseph_ufl  4096 Jul  8 12:25 bin
drwxrwxr-x  5 joseph_ufl joseph_ufl  4096 Jul  8 12:25 compute-sanitizer
-rw-r--r--  1 joseph_ufl joseph_ufl   160 Jul  8 12:25 DOCS
-rw-r--r--  1 joseph_ufl joseph_ufl 63021 Jul  8 12:25 EULA.txt
drwxrwxr-x  5 joseph_ufl joseph_ufl  4096 Jul  8 12:25 extras
drwxrwxr-x  5 joseph_ufl joseph_ufl  4096 Jul  8 12:25 gds
drwxrwxr-x  2 joseph_ufl joseph_ufl  4096 Jul  8 12:25 gds-12.6
lrwxrwxrwx  1 joseph_ufl joseph_ufl    28 Jul  8 12:25 include -> targets/x86_64-linux/include
lrwxrwxrwx  1 joseph_ufl joseph_ufl    24 Jul  8 12:25 lib64 -> targets/x86_64-linux/lib
drwxrwxr-x  7 joseph_ufl joseph_ufl  4096 Jul  8 12:25 libnvvp
drwxrwxr-x  7 joseph_ufl joseph_ufl  4096 Jul  8 12:25 nsight-compute-2024.3.2
drwxrwxr-x  2 joseph_ufl joseph_ufl  4096 Jul  8 12:25 nsightee_plugins
drwxrwxr-x  6 joseph_ufl joseph_ufl  4096 Jul  8 12:25 nsight-systems-2024.5.1
drwxrwxr-x  3 joseph_ufl joseph_ufl  4096 Jul  8 12:25 nvml
drwxrwxr-x  6 joseph_ufl joseph_ufl  4096 Jul  8 12:25 nvvm
drwxrwxr-x  2 joseph_ufl joseph_ufl  4096 Jul  8 12:25 pkgconfig
-rw-r--r--  1 joseph_ufl joseph_ufl   524 Jul  8 12:25 README
drwxrwxr-x  3 joseph_ufl joseph_ufl  4096 Jul  8 12:25 share
drwxrwxr-x  2 joseph_ufl joseph_ufl  4096 Jul  8 12:25 src
drwxrwxr-x  3 joseph_ufl joseph_ufl  4096 Jul  8 12:25 targets
drwxrwxr-x  2 joseph_ufl joseph_ufl  4096 Jul  8 12:25 tools
-rw-r--r--  1 joseph_ufl joseph_ufl  2935 Jul  8 12:25 version.json
--- bin
total 242252
drwxrwxr-x  3 joseph_ufl joseph_ufl     4096 Jul  8 12:25 .
drwxrwxr-x 18 joseph_ufl joseph_ufl     4096 Jul  8 12:25 ..
-rwxr-xr-x  1 joseph_ufl joseph_ufl    88848 Jul  8 12:25 bin2c
lrwxrwxrwx  1 joseph_ufl joseph_ufl        4 Jul  8 12:25 computeprof -> nvvp
-rwxr-xr-x  1 joseph_ufl joseph_ufl      112 Jul  8 12:25 compute-sanitizer
drwxrwxr-x  2 joseph_ufl joseph_ufl     4096 Jul  8 12:25 crt
-rwxr-xr-x  1 joseph_ufl joseph_ufl  7670488 Jul  8 12:25 cudafe++
-rwxr-xr-x  1 joseph_ufl joseph_ufl     1951 Jul  8 12:25 cuda-gdb
-rwxr-xr-x  1 joseph_ufl joseph_ufl 13757288 Jul  8 12:25 cuda-gdb-minimal
-rwxr-xr-x  1 joseph_ufl joseph_ufl 16040360 Jul  8 12:25 cuda-gdb-python3.10-tui
-rwxr-xr-x  1 joseph_ufl joseph_ufl 16040032 Jul  8 12:25 cuda-gdb-python3.11-tui
-rwxr-xr-x  1 joseph_ufl joseph_ufl 16048512 Jul  8 12:25 cuda-gdb-python3.12-tui
-rwxr-xr-x  1 joseph_ufl joseph_ufl 16040248 Jul  8 12:25 cuda-gdb-python3.8-tui
-rwxr-xr-x  1 joseph_ufl joseph_ufl 16040296 Jul  8 12:25 cuda-gdb-python3.9-tui
-rwxr-xr-x  1 joseph_ufl joseph_ufl   825936 Jul  8 12:25 cuda-gdbserver
-rwxr-xr-x  1 joseph_ufl joseph_ufl  1047075 Jul  8 12:25 cuda-uninstaller
-rwxr-xr-x  1 joseph_ufl joseph_ufl    75928 Jul  8 12:25 cu++filt
-rwxr-xr-x  1 joseph_ufl joseph_ufl   708112 Jul  8 12:25 cuobjdump
-rwxr-xr-x  1 joseph_ufl joseph_ufl  1245272 Jul  8 12:25 fatbinary
-rwxr-xr-x  1 joseph_ufl joseph_ufl     3826 Jul  8 12:25 ncu
-rwxr-xr-x  1 joseph_ufl joseph_ufl     3616 Jul  8 12:25 ncu-ui
-rwxr-xr-x  1 joseph_ufl joseph_ufl     1580 Jul  8 12:25 nsight_ee_plugins_manage.sh
-rwxr-xr-x  1 joseph_ufl joseph_ufl      197 Jul  8 12:25 nsight-sys
-rwxr-xr-x  1 joseph_ufl joseph_ufl      743 Jul  8 12:25 nsys
-rwxr-xr-x  1 joseph_ufl joseph_ufl      833 Jul  8 12:25 nsys-ui
-rwxr-xr-x  1 joseph_ufl joseph_ufl 23099816 Jul  8 12:25 nvcc
-rwxr-xr-x  1 joseph_ufl joseph_ufl    10456 Jul  8 12:25 __nvcc_device_query
-rw-r--r--  1 joseph_ufl joseph_ufl      425 Jul  8 12:25 nvcc.profile
-rwxr-xr-x  1 joseph_ufl joseph_ufl 50683896 Jul  8 12:25 nvdisasm
-rwxr-xr-x  1 joseph_ufl joseph_ufl 31368744 Jul  8 12:25 nvlink
-rwxr-xr-x  1 joseph_ufl joseph_ufl  6030656 Jul  8 12:25 nvprof
-rwxr-xr-x  1 joseph_ufl joseph_ufl   113664 Jul  8 12:25 nvprune
-rwxr-xr-x  1 joseph_ufl joseph_ufl      285 Jul  8 12:25 nvvp
-rwxr-xr-x  1 joseph_ufl joseph_ufl 31035152 Jul  8 12:25 ptxas
--- nvvm
/mnt/nvme10/joseph_ufl/cuda-12.6.3/nvvm:
total 24
drwxrwxr-x  6 joseph_ufl joseph_ufl 4096 Jul  8 12:25 .
drwxrwxr-x 18 joseph_ufl joseph_ufl 4096 Jul  8 12:25 ..
drwxrwxr-x  2 joseph_ufl joseph_ufl 4096 Jul  8 12:25 bin
drwxrwxr-x  2 joseph_ufl joseph_ufl 4096 Jul  8 12:25 include
drwxrwxr-x  2 joseph_ufl joseph_ufl 4096 Jul  8 12:25 lib64
drwxrwxr-x  2 joseph_ufl joseph_ufl 4096 Jul  8 12:25 libdevice

/mnt/nvme10/joseph_ufl/cuda-12.6.3/nvvm/bin:
total 33460
drwxrwxr-x 2 joseph_ufl joseph_ufl     4096 Jul  8 12:25 .
drwxrwxr-x 6 joseph_ufl joseph_ufl     4096 Jul  8 12:25 ..
-rwxr-xr-x 1 joseph_ufl joseph_ufl 34254632 Jul  8 12:25 cicc
--- lib64 / targets
cmake
libaccinj64.so
libaccinj64.so.12.6
libaccinj64.so.12.6.80
libcublasLt.so
[... 24 lines trimmed: libcublasLt, libcublas, libcudadevrt.a, libcudart (.so.12.6.77 and _static.a), libcufft, libcufftw, libcufile_rdma ...]
libcufile_rdma_static.a
x86_64-linux
--- file nvcc
/mnt/nvme10/joseph_ufl/cuda-12.6.3/bin/nvcc: ELF 64-bit LSB executable, x86-64, version 1 (SYSV), dynamically linked, interpreter /lib64/ld-linux-x86-64.so.2, for GNU/Linux 2.6.32, stripped
--- version.json
{
   "cuda" : {
      "name" : "CUDA SDK",
      "version" : "12.6.20241114"
   },
   "cuda_cccl" : {
      "name" : "CUDA C++ Core Compute Libraries",
      "version" : "12.6.77"
   },
   "cuda_cudart" : {
      "name" : "CUDA Runtime (cudart)",
      "version" : "12.6.77"
   },
   "cuda_cuobjdump" : {
      "name" : "cuobjdump",
      "version" : "12.6.77"
   },
[... 20 lines trimmed: cuda_cupti, cuda_cuxxfilt, cuda_demo_suite, cuda_gdb, cuda_nsight ...]
   "cuda_nvcc" : {
      "name" : "CUDA NVCC",
      "version" : "12.6.85"
   },
[... remaining entries trimmed; head -c 1500 cut the file inside "cuda_opencl" ...]
--- du
7.0G	/mnt/nvme10/joseph_ufl/cuda-12.6.3

[rx] id=20260923-041407-exec-df26 rc=0 state=done
```

`ls -la` omits the year for recent files. By that convention, "Jul 8 12:25" means July 8 of the current year.

#### 3e. NVHPC directory names, deeper, and other mounts

```
uv run tools/rx.py exec --timeout 500 -- 'date -Is; echo "--- / -xdev maxdepth 5 dirs"; timeout 180 find / -xdev -maxdepth 5 -type d \( -name hpc_sdk -o -iname "nvhpc*" -o -name "cuda-1*" -o -name Linux_x86_64 \) 2>/dev/null | head -20; echo "find-root rc=${PIPESTATUS[0]}"; echo "--- /mnt/nvme10 maxdepth 4 dirs"; timeout 180 find /mnt/nvme10 -maxdepth 4 -type d \( -name hpc_sdk -o -iname "nvhpc*" -o -name Linux_x86_64 \) 2>/dev/null | head -20; echo "find-nvme10 rc=${PIPESTATUS[0]}"; echo "--- other /mnt mounts maxdepth 4"; for m in nvme0 nvme1 nvme2 nvme3 nvme7 llama; do timeout 60 find /mnt/$m -maxdepth 4 \( -name nvcc -o -name nvc++ -o -name hpc_sdk -o -iname "nvhpc*" \) 2>/dev/null | head -5; echo "$m rc=${PIPESTATUS[0]}"; done'
```

```
2026-09-23T04:14:44-07:00
--- / -xdev maxdepth 5 dirs
find-root rc=1
--- /mnt/nvme10 maxdepth 4 dirs
find-nvme10 rc=1
--- other /mnt mounts maxdepth 4
nvme0 rc=1
nvme1 rc=1
nvme2 rc=1
nvme3 rc=1
nvme7 rc=1
llama rc=0

[rx] id=20260923-041444-exec-31c7 rc=0 state=done
```

No NVHPC directory and no `nvc++` binary turned up anywhere searched.

#### 3f. Host facts for P0.7

```
uv run tools/rx.py exec -- 'date -Is; uname -a; echo "---"; cat /etc/os-release; echo "---"; command -v gcc g++; gcc --version; g++ --version | head -1; ls /usr/bin/gcc-* /usr/bin/g++-* 2>/dev/null; echo "---"; ldd --version | head -1; getconf GNU_LIBC_VERSION; echo "--- cpu"; lscpu | grep -E "^(Architecture|Model name|CPU\(s\)):"'
```

```
2026-09-23T04:14:53-07:00
Linux alpha01 6.6.29+main+3.0.0r1-amd64-gio-epilmore-dev+ #1.914760.gio SMP PREEMPT_DYNAMIC Wed Jun 26 13:20:04 PDT 2024 x86_64 x86_64 x86_64 GNU/Linux
---
PRETTY_NAME="Ubuntu 22.04.5 LTS"
NAME="Ubuntu"
VERSION_ID="22.04"
VERSION="22.04.5 LTS (Jammy Jellyfish)"
VERSION_CODENAME=jammy
ID=ubuntu
ID_LIKE=debian
HOME_URL="https://www.ubuntu.com/"
SUPPORT_URL="https://help.ubuntu.com/"
BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"
PRIVACY_POLICY_URL="https://www.ubuntu.com/legal/terms-and-policies/privacy-policy"
UBUNTU_CODENAME=jammy
---
/usr/bin/gcc
/usr/bin/g++
gcc (Ubuntu 12.3.0-1ubuntu1~22.04.3) 12.3.0
Copyright (C) 2022 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

g++ (Ubuntu 12.3.0-1ubuntu1~22.04.3) 12.3.0
/usr/bin/g++-11
/usr/bin/g++-12
/usr/bin/gcc-11
/usr/bin/gcc-12
/usr/bin/gcc-ar
/usr/bin/gcc-ar-11
/usr/bin/gcc-ar-12
/usr/bin/gcc-nm
/usr/bin/gcc-nm-11
/usr/bin/gcc-nm-12
/usr/bin/gcc-ranlib
/usr/bin/gcc-ranlib-11
/usr/bin/gcc-ranlib-12
---
ldd (Ubuntu GLIBC 2.35-0ubuntu3.15) 2.35
glibc 2.35
--- cpu
Architecture:                         x86_64
CPU(s):                               256
Model name:                           AMD EPYC 9534 64-Core Processor

[rx] id=20260923-041453-exec-1aa9 rc=0 state=done
```

### 4. nvcc cross-compile for sm_80, and diagnostic captures

The command was `uv run tools/rx.py exec --timeout 600 --tail 400 -- "$(cat nvcc_probe.sh)"`. The script was written in a local scratch directory, not in the repository, and is passed as one string. Its text, verbatim:

```
date -Is
echo "LANG=${LANG:-unset} LC_ALL=${LC_ALL:-unset}"
C=/mnt/nvme10/joseph_ufl/cuda-12.6.3
export PATH="$C/bin:$PATH"
mkdir -p "$TMPDIR"
W=$(mktemp -d "$TMPDIR/p0-nvcc.XXXXXX")
cd "$W" || exit 1
echo "workdir=$W"
echo "--- which"
command -v nvcc cuobjdump; echo "command-v rc=$?"
echo "--- nvcc --version"
nvcc --version; echo "nvcc-version rc=$?"
echo "--- nvcc.profile"
cat "$C/bin/nvcc.profile"
cat > t.cu <<'EOF'
#include <cstdio>
__global__ void add(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}
int main() {
  const int n = 1024;
  float *a, *b, *c;
  cudaMalloc(&a, n * sizeof(float));
  cudaMalloc(&b, n * sizeof(float));
  cudaMalloc(&c, n * sizeof(float));
  add<<<(n + 255) / 256, 256>>>(a, b, c, n);
  cudaError_t e = cudaDeviceSynchronize();
  std::printf("%s\n", cudaGetErrorString(e));
  cudaFree(a); cudaFree(b); cudaFree(c);
  return 0;
}
EOF
echo "--- compile t.cu (sm_80, bible flags)"
nvcc -std=c++14 -Xcompiler -Wall -arch=sm_80 -O3 -o t t.cu 2> t.stderr; echo "nvcc rc=$?"
echo "stderr bytes: $(wc -c < t.stderr)"; cat t.stderr
ls -l t; file t
echo "--- cuobjdump --list-elf t"
cuobjdump --list-elf t; echo "list-elf rc=$?"
echo "--- cuobjdump --list-ptx t"
cuobjdump --list-ptx t; echo "list-ptx rc=$?"
echo "--- cuobjdump -sass t (head)"
cuobjdump -sass t | head -12
echo "--- ldd t"
ldd t
cat > undef.cu <<'EOF'
__global__ void saxpy(int n, float a, const float* x, float* y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = a * x[i] + undefined_var;
}
int main() { return 0; }
EOF
echo "--- compile undef.cu"
nvcc -std=c++14 -Xcompiler -Wall -arch=sm_80 -O3 -o undef undef.cu 2> undef.stderr; echo "nvcc rc=$?"
echo "stderr bytes: $(wc -c < undef.stderr)"
echo "<<<cat -A undef.stderr"; cat -A undef.stderr; echo ">>>"
cat > unused.cu <<'EOF'
__global__ void scale(float* x, int n) {
  float unused = 0.0f;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] *= 2.0f;
}
int main() {
  int unused_host = 0;
  return 0;
}
EOF
echo "--- compile unused.cu"
nvcc -std=c++14 -Xcompiler -Wall -arch=sm_80 -O3 -o unused unused.cu 2> unused.stderr; echo "nvcc rc=$?"
echo "stderr bytes: $(wc -c < unused.stderr)"
echo "<<<cat -A unused.stderr"; cat -A unused.stderr; echo ">>>"
echo "--- cleanup"
cd "$TMPDIR" && rm -rf "$W"; echo "cleanup rc=$?"; ls -d "$W" 2>&1
```

Output (full):

```
2026-09-23T04:16:03-07:00
LANG=en_US.UTF-8 LC_ALL=unset
workdir=/mnt/nvme10/joseph_ufl/tmp/p0-nvcc.AxWvOu
--- which
/mnt/nvme10/joseph_ufl/cuda-12.6.3/bin/nvcc
/mnt/nvme10/joseph_ufl/cuda-12.6.3/bin/cuobjdump
command-v rc=0
--- nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Tue_Oct_29_23:50:19_PDT_2024
Cuda compilation tools, release 12.6, V12.6.85
Build cuda_12.6.r12.6/compiler.35059454_0
nvcc-version rc=0
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
--- compile t.cu (sm_80, bible flags)
nvcc rc=0
stderr bytes: 0
-rwxrwxr-x 1 joseph_ufl joseph_ufl 1003232 Sep 23 04:16 t
t: ELF 64-bit LSB pie executable, x86-64, version 1 (SYSV), dynamically linked, interpreter /lib64/ld-linux-x86-64.so.2, BuildID[sha1]=ea97187c89986cf80807f4592e2329c7bbe51397, for GNU/Linux 3.2.0, not stripped
--- cuobjdump --list-elf t
ELF file    1: t.1.sm_80.cubin
ELF file    2: t.2.sm_80.cubin
list-elf rc=0
--- cuobjdump --list-ptx t
PTX file    1: t.1.sm_80.ptx
list-ptx rc=0
--- cuobjdump -sass t (head)

Fatbin elf code:
================
arch = sm_80
code version = [1,7]
host = linux
compile_size = 64bit

        code for sm_80

Fatbin elf code:
================
--- ldd t
        linux-vdso.so.1 (0x00007ffe25935000)
        libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007fbe89c00000)
        /lib64/ld-linux-x86-64.so.2 (0x00007fbe89f62000)
--- compile undef.cu
nvcc rc=2
stderr bytes: 177
<<<cat -A undef.stderr
undef.cu(4): error: identifier "undefined_var" is undefined$
      y[i] = a * x[i] + undefined_var;$
                        ^$
$
1 error detected in the compilation of "undef.cu".$
>>>
--- compile unused.cu
nvcc rc=0
stderr bytes: 322
<<<cat -A unused.stderr
unused.cu(2): warning #177-D: variable "unused" was declared but never referenced$
    float unused = 0.0f;$
          ^$
$
Remark: The warnings can be suppressed with "-diag-suppress <warning-number>"$
$
unused.cu(7): warning #177-D: variable "unused_host" was declared but never referenced$
    int unused_host = 0;$
        ^$
$
>>>
--- cleanup
cleanup rc=0
ls: cannot access '/mnt/nvme10/joseph_ufl/tmp/p0-nvcc.AxWvOu': No such file or directory

[rx] id=20260923-041603-exec-cec4 rc=2 state=failed
```

The rx `rc=2 state=failed` comes only from the final `ls -d "$W"`, which is the removal check: it fails because the work directory is gone. Every compile step reports its own rc above. The produced binaries were not run.

### 5. Cubin contents and host GCC warning under two locales

The command was `uv run tools/rx.py exec --timeout 600 --tail 400 -- "$(cat nvcc_probe2.sh)"`. The script text, verbatim:

```
date -Is
C=/mnt/nvme10/joseph_ufl/cuda-12.6.3
export PATH="$C/bin:$PATH"
mkdir -p "$TMPDIR"
W=$(mktemp -d "$TMPDIR/p0-nvcc.XXXXXX")
cd "$W" || exit 1
echo "workdir=$W"
cat > t.cu <<'EOF'
[same t.cu as step 4]
EOF
nvcc -std=c++14 -Xcompiler -Wall -arch=sm_80 -O3 -o t t.cu; echo "nvcc rc=$?"
echo "--- cuobjdump -sass t: arch and function lines"
cuobjdump -sass t | grep -E "arch =|Function :"; echo "grep rc=${PIPESTATUS[1]}"
cat > signcmp.cu <<'EOF'
#include <cstddef>
int count(const int* v, std::size_t n) {
  int s = 0;
  for (int i = 0; i < n; i++) {
    s += v[i];
  }
  return s;
}
int main() {
  int v[4] = {1, 2, 3, 4};
  return count(v, 4) == 10 ? 0 : 1;
}
EOF
echo "--- compile signcmp.cu (default env)"
nvcc -std=c++14 -Xcompiler -Wall -arch=sm_80 -O3 -o signcmp signcmp.cu 2> signcmp.stderr; echo "nvcc rc=$?"
echo "stderr bytes: $(wc -c < signcmp.stderr)"
echo "<<<cat -A signcmp.stderr"; cat -A signcmp.stderr; echo ">>>"
echo "--- compile signcmp.cu (LC_ALL=C)"
LC_ALL=C nvcc -std=c++14 -Xcompiler -Wall -arch=sm_80 -O3 -o signcmp signcmp.cu 2> signcmp_c.stderr; echo "nvcc rc=$?"
echo "stderr bytes: $(wc -c < signcmp_c.stderr)"
echo "<<<cat -A signcmp_c.stderr"; cat -A signcmp_c.stderr; echo ">>>"
echo "--- cleanup"
cd "$TMPDIR" && rm -rf "$W"; echo "cleanup rc=$?"; test ! -e "$W" && echo "workdir removed"
```

(The `[same t.cu as step 4]` line stands in for the 17 source lines sent verbatim, identical to step 4.)

Output (full):

```
2026-09-23T04:16:32-07:00
workdir=/mnt/nvme10/joseph_ufl/tmp/p0-nvcc.Opx67V
nvcc rc=0
--- cuobjdump -sass t: arch and function lines
arch = sm_80
arch = sm_80
                Function : _Z3addPKfS0_Pfi
arch = sm_80
grep rc=0
--- compile signcmp.cu (default env)
nvcc rc=0
stderr bytes: 300
<<<cat -A signcmp.stderr
signcmp.cu: In function M-bM-^@M-^Xint count(const int*, std::size_t)M-bM-^@M-^Y:$
signcmp.cu:4:19: warning: comparison of integer expressions of different signedness: M-bM-^@M-^XintM-bM-^@M-^Y and M-bM-^@M-^Xstd::size_tM-bM-^@M-^Y {aka M-bM-^@M-^Xlong unsigned intM-bM-^@M-^Y} [-Wsign-compare]$
    4 |   for (int i = 0; i < n; i++) {$
      |                 ~~^~~$
>>>
--- compile signcmp.cu (LC_ALL=C)
nvcc rc=0
stderr bytes: 284
<<<cat -A signcmp_c.stderr
signcmp.cu: In function 'int count(const int*, std::size_t)':$
signcmp.cu:4:19: warning: comparison of integer expressions of different signedness: 'int' and 'std::size_t' {aka 'long unsigned int'} [-Wsign-compare]$
    4 |   for (int i = 0; i < n; i++) {$
      |                 ~~^~~$
>>>
--- cleanup
cleanup rc=0
workdir removed

[rx] id=20260923-041632-exec-6b85 rc=0 state=done
```

The byte counts are consistent with the quote escaping: eight quotes, each 3 bytes in UTF-8 against 1 in ASCII, add 16 bytes, and 300 - 284 = 16.

### 6. nvc++

Not run. `nvc++` does not exist on alpha01 (steps 2, 3a, 3b, 3e), so there is no `--version`, no `-mp=gpu -gpu=cc80` test, and no stderr capture.

## Sources

Accessed 2026-09-23. These are documentation, not measurements.

- NVIDIA HPC SDK 24.11 Release Notes, https://docs.nvidia.com/hpc-sdk/archive/24.11/hpc-sdk-release-notes/index.html ("Last updated November 13, 2024"):
  - "The NVIDIA HPC SDK 24.11 includes the following CUDA toolchain versions: CUDA 11.8" and "CUDA 12.6u2".
  - Ubuntu 22.04 is a supported x86_64 distribution.
  - Minimum gcc/glibc toolchain: C++17 7.5, C++20 10.1, C++23 12.1.
- NVIDIA HPC Compilers Reference Guide 24.11, `-gpu` option, https://docs.nvidia.com/hpc-sdk/archive/24.11/compilers/hpc-compilers-ref-guide/index.html:
  - ccXY: "Generate code for a device with compute capability X.Y."
  - ccnative: "Detects the visible GPUs on the system and generates codes for them. If no device is available, the compute capability matching NVCC's default will be used."
- NVIDIA HPC Compilers User's Guide 24.11, https://docs.nvidia.com/hpc-sdk/archive/24.11/compilers/hpc-compilers-user-guide/index.html: "Use `-mp=gpu` to parallelize OpenMP regions for offload to an NVIDIA GPU"; `-mp` defaults to multicore CPU parallelization.
- CUDA Installation Guide for Linux 12.6 (12.6.2 archive), https://docs.nvidia.com/cuda/archive/12.6.2/cuda-installation-guide-linux/index.html:
  - Supported host GCC on x86_64: 6.x - 13.2.
  - Ubuntu 22.04 row: default GCC 12.3.0, GLIBC 2.35.
  - The runfile supports `--silent` and `--toolkitpath=<path>`, and installs without sudo into a directory the user can write.

## Findings

Provenance for every [MEASURED] item:

- Host: alpha01, Ubuntu 22.04.5 LTS, kernel 6.6.29+main+3.0.0r1-amd64-gio-epilmore-dev+, AMD EPYC 9534 (256 CPUs).
- Toolchain: CUDA 12.6.3 (`version.json` CUDA SDK 12.6.20241114, nvcc V12.6.85, build cuda_12.6.r12.6/compiler.35059454_0), host GCC 12.3.0, glibc 2.35.
- Device: none. The runs were CPU only, and no GPU was opened or queried.
- Date: 2026-09-23. Local commit: 22f2215.

1. Compilers present:
   - nvcc: present but not on PATH [MEASURED]. It is at `/mnt/nvme10/joseph_ufl/cuda-12.6.3/bin/nvcc`, "Cuda compilation tools, release 12.6, V12.6.85". The tree is a complete CUDA 12.6.3 toolkit in runfile layout (ptxas, nvlink, fatbinary, cudafe++, nvvm/bin/cicc, cuobjdump, cudart, and `cuda-uninstaller`), 7.0G by `du -sh`, with files dated Jul 8 (year omitted by `ls`) and owner joseph_ufl.
   - nvc++: absent [MEASURED]. It is not on PATH, and no `nvc++` binary, `hpc_sdk` directory, or `nvhpc*` directory exists anywhere searched: `/` to depth 5, `/opt` to depth 7, `/mnt/nvme10` to depth 5 by name and depth 4 by directory, other data mounts to depth 4, dpkg, the loader cache, and environment modules.
   - Nothing else: there is no system CUDA install (no `/usr/local/cuda*`, no CUDA dpkg packages, no `libcudart` or `libcuda.so` in the loader cache), and `$LASSI_TOOLCHAINS` is empty [MEASURED].
2. nvcc cross-compiles for sm_80 without a GPU: yes [MEASURED].
   - The bible's flags `nvcc -std=c++14 -Xcompiler -Wall -arch=sm_80 -O3` built a kernel plus host main with exit 0 and empty stderr.
   - `cuobjdump --list-elf` shows two sm_80 cubins, and `--list-ptx` shows one sm_80 PTX.
   - `cuobjdump -sass` shows `_Z3addPKfS0_Pfi` compiled for sm_80. The second cubin lists no function; its origin was not investigated.
   - The binary links only libc dynamically (cudart is static), so building needed no driver library.
   - Evidence that no NVIDIA GPU or driver is present: the loader cache has no `libcuda.so` [MEASURED], and the bible's Environment State (2026-09-22) records no NVIDIA GPU on I/ONX. No device command was run.
3. nvc++ cross-compile for cc80: not testable, because the compiler is absent. Inference, not measured: the HPC compilers reference documents `-gpu=ccXY` as generating code for the named compute capability, and `ccnative` as falling back when no device is visible. So `-gpu=cc80` is expected to build without a GPU. P0.7 must verify it with the same kind of test.
4. Free space [MEASURED]:
   - `/mnt/nvme10`: 513G available of 3.5T (86% used) by `df -h`. `rx doctor` reports `scratch_free_gb` 550.6, the same quantity in decimal GB.
   - The root filesystem has 300G available (82% used) by `df -h`, and `rx doctor` reports `root_free_gb` 321.3. This differs from the bible's Environment State line "Root filesystem full". Rule 7 applies either way.
5. Diagnostics for P0.15 [MEASURED; evidence only, no test was edited]:
   - An EDG front-end error makes nvcc exit 2. The format is `file(line): error: ...`, then a source echo, a caret line, a blank line, and `1 error detected in the compilation of "file".`
   - EDG `warning #177-D` fires for unused variables in both device and host code. The `Remark: ... -diag-suppress ...` line follows only the first warning. Each echo and caret block ends with a blank line.
   - GCC `-Wunused-variable` did not appear for the unused host variable in this build; only EDG's #177-D did.
   - Host GCC warnings pass through with the original `.cu` file name, `file:line:col`, and the `[-Wsign-compare]` flag.
   - The gate's default environment is `LANG=en_US.UTF-8`, so GCC quotes are UTF-8 U+2018 and U+2019. With `LC_ALL=C` they are ASCII `'`, which matches what the fixture README assumes.

## Consequences for the Plan

- Bible, Execution Backends, executor `none`: the Status "Available" does not hold as written. Its Key settings name both nvcc and nvc++, and nvc++ is absent. The Status becomes partial until P0.7 installs NVHPC; then it can return to "Available" with P0.7's evidence.
- Bible, Host Facts: "CUDA 12.6.3 headers under the corpus pipeline's project root" understates the asset. It is a complete CUDA 12.6.3 toolkit at `/mnt/nvme10/joseph_ufl/cuda-12.6.3`. Inference: this is the same tree the bible refers to. The measured facts are nvcc V12.6.85 in a directory named cuda-12.6.3 (the release label comes from that name) and a path under the working root; whether that path counts as the corpus pipeline's project root was not checked.
- P0.7, CUDA. The existing tree is usable, but it is a user install of unknown install history outside the `$LASSI_TOOLCHAINS/<name>@<pin>` convention in AGENTS.md, and the v0 corpus pipeline may depend on it. P0.7's "pin usable system installs instead" clause does not cleanly cover it, because it is not a system install. There are two compliant ways forward:
  - (a) Recommended: `toolchains/cuda.sh` installs the same release (CUDA 12.6.3 runfile, toolkit only) under `$LASSI_TOOLCHAINS/cuda@12.6.3`. The install stays scripted, idempotent, and reproducible, and matches the existing tree's version. The measured size of the equivalent existing tree is 7.0G, well within the 513G free.
  - (b) Pin the existing tree in place: record its path and version in `toolchains/cuda.pin`, and have the script verify rather than install. This is cheaper, but it depends on an unmanaged tree.

  Either way the pin is P0.7's Decision Log entry.
- P0.7, NVHPC: an install is required, because nothing usable exists. The bible says only "NVHPC 2024". Inference from the release notes: 24.11 bundles CUDA 12.6u2 and supports Ubuntu 22.04 with host GCC 12.3 and glibc 2.35, which makes it the 2024 release closest to the CUDA 12.6.3 nvcc. The exact release is P0.7's pin decision. Its size is not measured here; P0.7 measures it. The measured free space (513G on `/mnt/nvme10`) is far above the gate's big-job floor (60 GB).
- P0.5 and P0.15 (recommendation, not measured): the executor or toolchain adapter should run compilers with `LC_ALL=C`. Otherwise the parser must accept UTF-8 quotes, and stored diagnostics must be ASCII-safe before they enter any committed artifact. P0.15 can capture its nvcc fixtures now with `/mnt/nvme10/joseph_ufl/cuda-12.6.3`, or after P0.7 with the pinned path. The nvc++ fixtures must wait for P0.7.

## Confidence

- High that nvcc 12.6.85 exists at the recorded path and builds sm_80 code with the bible's flags on a host without a GPU. This is direct output, reproduced in two separate runs.
- High that nvc++ is absent from every searched location. The search was bounded (depth limits, 60-240 s timeouts), so an install buried deeper in another tenant's mount cannot be ruled out. Such an install would not be pinnable anyway.
- The nvc++ cc80 cross-compile is inference from documentation only.

## Applied Bible Edit (master revision 47)

The edit below was applied to the master copy (revisions 44 to 47 are this edit alone, checked with a view of every change since revision 43) and mirrored in docs/BIBLE.md.

- Execution Backends, executor `none`, Status: "Partial: nvcc 12.6 builds sm_80 without a GPU [MEASURED 2026-09-23]; nvc++ absent until P0.7 installs NVHPC 2024".
- Host Facts, last bullet: keeps "CUDA 12.6.3 headers under the corpus pipeline's project root" (whether that is the same tree was not verified) and adds the complete CUDA 12.6 toolkit (directory `cuda-12.6.3`, nvcc V12.6.85) at `/mnt/nvme10/joseph_ufl/cuda-12.6.3`, that no nvc++ or NVHPC install was found (bounded search), and host GCC, glibc, and locale, each tagged [MEASURED 2026-09-23].
- Decision Log, new top row dated 2026-09-23, and the count sentence.

The "12.6.3" label comes from the directory name; the tools print nvcc V12.6.85, cudart 12.6.77, and `version.json` CUDA SDK 12.6.20241114, which are consistent with it.

Not edited: the Environment State Storage row still says "Root filesystem full" (dated 2026-09-22), while this spike measured 300G available on `/` (finding 4). That row is left for the owner.
