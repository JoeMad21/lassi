# Spike: the workstation as LASSI's only host (capacity, tools, sandbox)

- Asked by: the owner's direction of 2026-10-15 (plans/LESSONS.md, Owner directions): "Tenstorrent cards are still down. Hardware is unavailable, including Furiosa. Do all work locally for the forseeable future. Plan around hardware unavailability." No STATUS task; this spike feeds the next planning step.
- Bible: Environment State (its table and Host Facts), Sandbox, Toolchain Pins, Build Roadmap (P5, P17), Agent Rules 6, 7, 10, and 17.
- Date: 2026-10-15, 00:05 to 00:25 EDT (UTC-04:00), on the workstation, with a second round of read-only probes at 01:09 to 01:11 (section "Probes of the planning round (2026-10-15)"). Windows last booted at 2026-10-14 13:44:42 EDT, so it had been up 10 h 38 min at 00:23 (that section). WSL's uptime in the first round was not logged and cannot be re-read: the distro had stopped by 01:09, and the second round restarted it.
- Base: branch p5-ir at 422876c, clean tree.
- Remote: none. No rx command, ssh, scp, rsync, or sftp; nothing touched alpha01. No Tenstorrent or RNGD device was named or queried.
- Device: the only device query was a read-only `nvidia-smi` on the workstation's own GPU (inventory: name, memory, driver).
- Writes: nothing installed, configured, or deleted. The probes wrote only to the session scratch directory outside the repository (the extracted setup script, its variants, and two probe scripts, all quoted below) and inside private mount namespaces (tmpfs mounts that vanish with the namespace). systemd-run created two transient user scopes (the scope probe and S4), which ended with their commands. The second round wrote nothing; its first WSL probe started the stopped distro.
- Shell: Git Bash on Windows. Windows probes ran as `powershell.exe -NoProfile -Command "<cmd>"`; WSL probes ran as `MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 --exec bash -lc '<cmd>'` (plans/LESSONS.md, Git on the workstation). Outputs below have CR bytes stripped and are trimmed where `[... trimmed ...]` says so.
- Labels: [MEASURED 2026-10-15, workstation] marks a value read by a probe in this spike. (inferred) marks reasoning from those values, source, or documentation. (doc) marks vendor documentation. (exploratory) marks a run of an uncommitted variant of repository code; it is not a measurement of the repository. (log) marks a value read from an existing file on the workstation that no LASSI run wrote.

## Question

Can the workstation carry LASSI's builds and runs while alpha01 and every accelerator are unavailable, and what does it lack? Five parts:

1. Windows: CPU, cores and threads, memory, drives and free space, OS build, GPU.
2. WSL: distros and versions, CPUs, memory and swap, .wslconfig limits, free space on the root and the mounts.
3. Compilers and build tools in WSL and on Windows, with exact versions, against the pins.
4. What lassi/executors/sandbox.py needs: unshare and user namespaces, the setup's mounts (LESSONS records `mount --move` exit 16 in WSL), the systemd user manager and cgroup limits, and core handling.
5. Whether the repository's Linux-only tests run in WSL today.

Why it matters: P5.4 (OWNER, OQ-054) and everything after it in P5 (P5.5 to P5.10, P5.12, P5.G) build and run tt-mlir, Polygeist, tt-metal, and ttsim, all planned for alpha01. Compiles of generated sources and native and ttsim program runs go through the sandbox (bible, Sandbox), so no local trial can run unless the sandbox works in WSL. A Windows test pass does not cover the Linux-only tests (plans/LESSONS.md, P17.10). P17.12, P17.13, and P17.G wait for a GPU host (OQ-040).

Classification: factual for the inventory, which read-only probes answer. Its consequences hold choices (installs, a WSL memory setting, new or changed pins, whether the workstation GPU is a GPU host), which go to the owner queue (drafts at the end).

## 1. Windows

```
$ powershell.exe -NoProfile -Command "Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed,VirtualizationFirmwareEnabled | Format-List; Get-CimInstance Win32_ComputerSystem | Select-Object Manufacturer,Model,TotalPhysicalMemory,NumberOfProcessors | Format-List; Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber,OSArchitecture,TotalVisibleMemorySize,FreePhysicalMemory,TotalVirtualMemorySize,FreeVirtualMemory | Format-List"
Name                          : Intel(R) Core(TM) i7-10700K CPU @ 3.80GHz
NumberOfCores                 : 8
NumberOfLogicalProcessors     : 16
MaxClockSpeed                 : 3792
VirtualizationFirmwareEnabled : False
Manufacturer        : Micro-Star International Co., Ltd.
Model               : MS-7C80
TotalPhysicalMemory : 34273165312
NumberOfProcessors  : 1
Caption                : Microsoft Windows 10 Home
Version                : 10.0.19045
BuildNumber            : 19045
OSArchitecture         : 64-bit
TotalVisibleMemorySize : 33469888
FreePhysicalMemory     : 11859540
TotalVirtualMemorySize : 80721344
FreeVirtualMemory      : 35708332

$ cmd.exe //c ver; powershell.exe ... Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion' ...
Microsoft Windows [Version 10.0.19045.7725]
ProductName    : Windows 10 Home
DisplayVersion : 22H2
CurrentBuild   : 19045
UBR            : 7725
EditionID      : Core

$ powershell.exe ... Get-Volume | Where-Object DriveLetter | Select-Object DriveLetter,FileSystemLabel,FileSystem,DriveType,HealthStatus,SizeGB,FreeGB
DriveLetter FileSystemLabel FileSystem DriveType HealthStatus SizeGB FreeGB
          C                 NTFS       Fixed     Healthy      3725.9  721.1
          Z EVO Volume      NTFS       Fixed     Healthy       931.5  578.3
          E S990Evo         NTFS       Fixed     Healthy        3726  313.9
          F PHD             NTFS       Fixed     Healthy      4657.5 1647.9
          D                 NTFS       Fixed     Healthy       931.5    387

$ powershell.exe ... Get-Partition | Where-Object DriveLetter ...; Get-Disk ...; Get-PhysicalDisk ...
DriveLetter DiskNumber SizeGB        Number FriendlyName            BusType
          C          3 3725.9             3 Samsung SSD 990 PRO 4TB NVMe
          E          2   3726             2 Samsung SSD 990 PRO 4TB NVMe
          F          4 4657.5             4 Seagate Portable        USB
          D          0  931.5             0 WDC WD10EZEX-00WN4A0    SATA   (MediaType HDD)
[Z: has no row in Get-Partition or Get-Disk; Get-PhysicalDisk lists a fifth disk, Samsung SSD 870 EVO 1TB on SATA, which matches Z:'s size and label (inferred)]

$ powershell.exe ... Win32_VideoController; registry HardwareInformation.qwMemorySize; nvidia-smi
Name          : NVIDIA GeForce RTX 3080
DriverVersion : 32.0.16.1742
NVIDIA GeForce RTX 3080 : 10 GiB
$ nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
name, memory.total [MiB], driver_version, compute_cap
NVIDIA GeForce RTX 3080, 10240 MiB, 617.42, 8.6
$ nvidia-smi | sed -n 3p
| NVIDIA-SMI 617.42                 KMD Version: 617.42        CUDA UMD Version: 13.4     |
```

[MEASURED 2026-10-15, workstation]: Intel Core i7-10700K, 8 cores and 16 threads, one socket; 31.9 GiB of RAM (34,273,165,312 bytes), 11.3 GiB free at the probe with WSL running; Windows 10 Home 22H2, build 19045.7725; NVIDIA GeForce RTX 3080 with 10,240 MiB, compute capability 8.6, driver 617.42 (CUDA user-mode 13.4). Free space: C: 721.1 GiB of 3725.9 (NVMe; holds the WSL disk, section 2), D: 387.0 of 931.5 (SATA HDD), E: 313.9 of 3726.0 (NVMe), F: 1647.9 of 4657.5 (USB HDD), Z: 578.3 of 931.5 (SATA SSD, inferred). VirtualizationFirmwareEnabled reads False while WSL 2 runs; Windows reports that when its own hypervisor is active (inferred), so it does not mean virtualization is off.

## 2. WSL

```
$ wsl.exe --version
WSL version: 3.0.3.0
Kernel version: 6.18.54.1-1
WSLg version: 1.0.79
[... trimmed: MSRDC, Direct3D, DXCore ...]
Windows version: 10.0.19045.7725
$ wsl.exe -l -v
  NAME              STATE           VERSION
* Ubuntu-24.04      Running         2
  docker-desktop    Stopped         2
$ od -c /c/Users/Joseph/.wslconfig
0000000  \r  \n
$ powershell.exe ... HKCU:\Software\Microsoft\Windows\CurrentVersion\Lxss (BasePath) and the ext4.vhdx sizes
Ubuntu-24.04 | C:\Users\Joseph\AppData\Local\wsl\{4808d599-62a1-44a2-8869-930ec79d0895} | v2   ext4.vhdx 16.2 GB
docker-desktop | \\?\C:\Users\Joseph\AppData\Local\Docker\wsl\main | v2                     ext4.vhdx  0.1 GB

$ wsl.exe -d Ubuntu-24.04 --exec bash -lc 'date -Is; head -4 /etc/os-release; uname -a; id; echo nproc=$(nproc); free -g; free -m; swapon --show; df -h / /mnt/*; lscpu | head -20'
2026-10-15T00:13:47-04:00
PRETTY_NAME="Ubuntu 24.04.2 LTS"
Linux DESKTOP-8R113EI 6.18.54.1-microsoft-standard-WSL2 #1 SMP PREEMPT_DYNAMIC Tue Oct  6 19:13:07 UTC 2026 x86_64 x86_64 x86_64 GNU/Linux
uid=1000(joe) gid=1000(joe) groups=1000(joe),4(adm),24(cdrom),27(sudo),30(dip),46(plugdev),100(users),1001(docker)
nproc=16
               total        used        free      shared  buff/cache   available
Mem:           15951         588       15352           4         208       15363
Swap:           4096           0        4096
NAME     TYPE      SIZE USED PRIO
/dev/sdc partition   4G   0B   -2
Filesystem      Size  Used Avail Use% Mounted on
/dev/sdd       1007G   15G  942G   2% /
C:\             3.7T  3.0T  722G  81% /mnt/c
D:\             932G  545G  388G  59% /mnt/d
E:\             3.7T  3.4T  314G  92% /mnt/e
F:\             4.6T  3.0T  1.7T  65% /mnt/f
none            7.8G  4.0K  7.8G   1% /mnt/wsl
none            7.8G   36K  7.8G   1% /mnt/wslg
Z:\             932G  354G  579G  38% /mnt/z
Model name:                              Intel(R) Core(TM) i7-10700K CPU @ 3.80GHz
Thread(s) per core:                      2
Core(s) per socket:                      8
Flags: [... trimmed ...] avx f16c [...] avx2 [...] (no avx512 flag)
Hypervisor vendor:                       Microsoft
$ wsl.exe -d Ubuntu-24.04 --exec bash -lc 'cat /etc/wsl.conf'
[boot]
systemd=true
[user]
default=joe
[network]
generateResolvConf = false
```

[MEASURED 2026-10-15, workstation]: WSL 3.0.3.0, kernel 6.18.54.1; two WSL 2 distros, Ubuntu-24.04 (24.04.2 LTS, the default, running) and docker-desktop (stopped). Ubuntu sees 16 CPUs, 15,951 MiB of memory, and 4,096 MiB of swap. After the restart of the second round, MemTotal read 16,334,740 kB, which is 15,951.9 MiB (section "Probes of the planning round (2026-10-15)"); the first round's MemTotal was not logged. %UserProfile%\.wslconfig holds only CRLF, so no limit is set and the documented defaults apply: memory 50% of Windows memory, processors all logical processors, swap 25% of memory rounded up to a GB (doc, learn.microsoft.com wsl-config). The values seen match those defaults. The WSL root is ext4 on a 1007G virtual disk with 942G free, but its VHDX is a growing file on C:, so the real ceiling is C:'s 721 GiB, shared with Windows (inferred). The CPU has AVX2 and no AVX-512, so -march=x86-64-v3 (tt-metal.pin HOST_FLAGS) runs here (inferred).

## 3. Tools

```
$ wsl.exe -d Ubuntu-24.04 --exec bash -lc 'for t in gcc g++ cc c++ clang clang++ clang-18 clang-19 clang-20 lld ld.lld ld.mold mold ld.gold ld cmake ninja make git python3 python3.10 python3.11 python3.12 python3.13 uv pip3 docker podman ccache gdb strace perl rustc cargo go java lit FileCheck mlir-opt llvm-config; do p=$(command -v $t 2>/dev/null); printf "%-12s %s\n" "$t" "${p:-MISSING}"; done'
gcc /usr/bin/gcc; g++ /usr/bin/g++; ld /usr/bin/ld; ld.gold /usr/bin/ld.gold; cmake /usr/bin/cmake; ninja /usr/bin/ninja;
make /usr/bin/make; git /usr/bin/git; python3 /usr/bin/python3; python3.12 /usr/bin/python3.12; pip3 /usr/bin/pip3; perl /usr/bin/perl;
docker /mnt/c/Program Files/Docker/Docker/resources/bin/docker
MISSING: clang clang++ clang-18 clang-19 clang-20 lld ld.lld ld.mold mold python3.10 python3.11 python3.13 uv podman ccache gdb strace rustc cargo go java lit FileCheck mlir-opt llvm-config
[... reformatted to fewer lines ...]

$ wsl.exe -d Ubuntu-24.04 --exec bash -lc 'gcc --version | head -1; g++ --version | head -1; ld --version | head -1; ld.gold --version | head -1; cmake --version | head -1; ninja --version; make --version | head -1; git --version; python3 --version; pip3 --version; ldd --version | head -1; mount --version; unshare --version; setpriv --version; prlimit --version; timeout --version | head -1; awk --version | head -1; systemctl --version | head -1; "/mnt/c/Program Files/Docker/Docker/resources/bin/docker" --version'
gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
GNU ld (GNU Binutils for Ubuntu) 2.42
GNU gold (GNU Binutils for Ubuntu 2.42) 1.16
cmake version 3.28.3
1.11.1
GNU Make 4.3
git version 2.43.0
Python 3.12.3
pip 24.0 from /usr/lib/python3/dist-packages/pip (python 3.12)
ldd (Ubuntu GLIBC 2.39-0ubuntu8.9) 2.39
mount from util-linux 2.39.3 (libmount 2.39.3: selinux, smack, btrfs, verity, namespaces, idmapping, statx, assert, debug)
unshare from util-linux 2.39.3
setpriv from util-linux 2.39.3
prlimit from util-linux 2.39.3
timeout (GNU coreutils) 9.4
GNU Awk 5.2.1, API 3.2, PMA Avon 8-g1, (GNU MPFR 4.2.1, GNU MP 6.3.0)
systemd 255 (255.4-1ubuntu8.17)
The command 'docker' could not be found in this WSL 2 distro.
We recommend to activate the WSL integration in Docker Desktop settings.
[... trimmed ...]

$ wsl.exe ... 'dpkg -l | grep -E "^ii +(gcc|g\+\+|clang|lld|mold|libomp|libgomp1|cmake|ninja-build|python3\.[0-9]+ |python3-venv|python3-dev|build-essential|uidmap|libseccomp)"'
build-essential 12.10ubuntu1; cmake 3.28.3-1build7; g++-13 13.3.0-6ubuntu2~24.04.1; gcc-13 13.3.0-6ubuntu2~24.04.1;
libgomp1:amd64 14.2.0-4ubuntu2~24.04.1; libseccomp2:amd64 2.5.5-1ubuntu3.1; ninja-build 1.11.1-2; python3-dev 3.12.3-0ubuntu2.1;
python3-venv 3.12.3-0ubuntu2.1; python3.12 3.12.3-1ubuntu0.17   [no clang, lld, mold, or libomp package]

$ (Git Bash) for t in uv python python3 py git gh cmake ninja gcc g++ clang clang++ cl docker podman wsl nvidia-smi nvcc make; do ...; done
uv, python, python3, py, git, gh, docker, wsl, nvidia-smi found; cmake ninja gcc g++ clang clang++ cl podman nvcc make MISSING
$ uv --version
uv 0.12.17 (635500036 2026-09-18 x86_64-pc-windows-msvc)
$ git --version; gh --version | head -1
git version 2.55.0.windows.3
gh version 2.101.0 (2026-09-15)
$ .venv/Scripts/python.exe --version; cat .venv/pyvenv.cfg
Python 3.10.21
home = C:\Users\Joseph\AppData\Roaming\uv\python\cpython-3.10-windows-x86_64-none
$ uv python list --only-installed   [a warning about a broken C:\ProgramData\chocolatey\bin\python3.13.exe shim trimmed]
cpython-3.14.2 (Python install manager and WindowsApps), cpython-3.12.14 (uv), cpython-3.10.21 (uv)
$ "/c/Program Files/Docker/Docker/resources/bin/docker.exe" --version
Docker version 28.0.4, build b8034c0
$ powershell.exe ... Get-Service com.docker.service
Status : Stopped   StartType : Manual
$ "/c/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe" -all -format text
[no output: no Visual Studio instance]
```

[MEASURED 2026-10-15, workstation], against the pins (toolchains/*.pin, read at 422876c; the pins were read on alpha01's Ubuntu 22.04):

| Tool | WSL Ubuntu-24.04 | Windows | Pin | Matches the pin |
| --- | --- | --- | --- | --- |
| C/C++ | gcc and g++ 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04.1); no g++-12 | none (no cl, gcc, or clang) | gcc.pin: /usr/bin/g++-12, "(Ubuntu 12.3.0-1ubuntu1~22.04.3) 12.3.0" | no |
| clang | none | none | tt-metal.pin: clang-20 "Ubuntu clang version 20.1.8 (++20250708082409+6fb913d3e2ec-1~exp1~20250708202428.132)"; tt-mlir.pin: clang-17 17.0.6 | no |
| Linker | GNU ld 2.42, gold 1.16; no lld, no mold | none | ld.lld-20 (LLD 20.1.8), ld.lld-17 (LLD 17.0.6) | no |
| CMake | 3.28.3 at /usr/bin/cmake | none | 3.31.6 at /usr/local/bin/cmake | no |
| Ninja | 1.11.1 at /usr/bin/ninja | none | 1.10.1 at /usr/bin/ninja | no |
| Python | 3.12.3 only (/usr/bin/python3.12) | 3.10.21 (uv-managed, the repo venv), 3.12.14, 3.14.2 | tt-mlir.pin: /usr/bin/python3.10 "Python 3.10.12"; the project: >=3.10,<3.11 | no in WSL; the repo venv is 3.10 on Windows |
| uv | none | 0.12.17 | (CI installs the lock's uv) | n/a |
| git | 2.43.0 | 2.55.0.windows.3 (gh 2.101.0) | none | n/a |
| Containers | none usable: `docker` is the Windows shim, which says WSL integration is off; no podman | Docker Desktop 28.0.4 client; com.docker.service Stopped (Manual) | none | n/a |
| GPU stack | /usr/lib/wsl/lib (libcuda, libnvidia-ml, ...); /dev/dxg; no /dev/nvidia* (section "Probes of the planning round (2026-10-15)") | driver 617.42, CUDA UMD 13.4; no nvcc | cuda.pin 12.6.3, nvhpc.pin 24.11 (alpha01 installs) | n/a |

Besides the versions, every install script refuses any root outside alpha01's scratch: toolchains/tt-mlir.sh:40-44, tt-metal.sh:44, and ttsim.sh:22 refuse anything outside /mnt/nvme10/joseph_ufl, and cuda.sh:20 and nvhpc.sh:14 anything outside /mnt/nvme10. Their check_host steps also refuse any host tool that differs from the pin (tt-mlir.sh:161-177, tt-metal.sh:147-156). So no pinned toolchain installs on this workstation as the scripts stand (inferred from source, not run).

An earlier, non-LASSI LLVM build sits in WSL (log): ~/mlir_dev/llvm-project at 60efb5c0a3f5 (llvmorg-23-init-10166, committed 2026-04-02), Release, LLVM_ENABLE_PROJECTS=clang;mlir, X86 only, Ninja, /usr/bin/c++ (CMAKE_CXX_COMPILER_VERSION 13.3.0), no LLVM_USE_LINKER, build tree 2.3G. Neither LASSI pin uses it (tt-mlir.pin LLVM_COMMIT 4efe170d; Polygeist's 26eb4285). Its .ninja_log (6,967 steps, written 2026-04-02 09:51 to 12:19) splits into three ninja invocations where the end times drop (script below):

```
invocation 1: steps=5079 wall_s=1231 summed_step_s=21291 mean_parallel=17.3
invocation 2: steps=8 wall_s=65 summed_step_s=248 mean_parallel=3.8
invocation 3: steps=1880 wall_s=5969 summed_step_s=11929 mean_parallel=2.0
total: invocations=3 wall_s=7265 summed_step_s=33468
```

(log) About 9.3 hours of summed step time; 2 h 01 min of ninja wall time in all. The first invocation ran at ninja's default parallelism for 16 CPUs and stopped after 5,079 steps; the last ran at a mean of 2.0. The log does not say why. A memory-limited failure in 15.6 GiB, followed by a lower -j, would fit, but that is not verified (inferred). Script (session scratch, ninjalog.awk):

```
NR == 1 { next }
{
  start = $1 + 0; end = $2 + 0
  if (sessions == 0 || end + 5000 < prevend) sessions++
  prevend = end
  steps[sessions]++
  if (end > wall[sessions]) wall[sessions] = end
  sum[sessions] += end - start
}
END { [prints the lines above] }
```

## 4. Sandbox prerequisites

What the sandbox needs (lassi/executors/sandbox.py docstring, lines 1 to 330, and the remote tests' host_problem, tests/executors/test_sandbox_remote.py:178-194): Linux; unshare, systemd-run, nice, setpriv, prlimit, timeout, python3, and awk on PATH; python3 in SANDBOX_PATH (/usr/sbin:/usr/bin:/sbin:/bin); a reachable user systemd manager ($XDG_RUNTIME_DIR/bus); mount_setattr (Linux 5.12 or later); x86_64 for the seccomp filter; unprivileged user namespaces; overlay with userxattr in a user namespace; and $LASSI_SCRATCH for the tests.

```
$ wsl.exe ... 'for f in /proc/sys/user/max_user_namespaces /proc/sys/user/max_mnt_namespaces /proc/sys/kernel/unprivileged_userns_clone /proc/sys/kernel/apparmor_restrict_unprivileged_userns /proc/sys/kernel/core_pattern /proc/sys/kernel/core_pipe_limit /proc/sys/kernel/yama/ptrace_scope /proc/sys/kernel/unprivileged_bpf_disabled /proc/sys/vm/unprivileged_userfaultfd; do ...; done; cat /sys/kernel/security/lsm; ulimit -Sc; ulimit -Hc; prlimit --core; zcat /proc/config.gz | grep ...'
/proc/sys/user/max_user_namespaces = 63769
/proc/sys/user/max_mnt_namespaces = 63769
/proc/sys/kernel/unprivileged_userns_clone: No such file or directory
/proc/sys/kernel/apparmor_restrict_unprivileged_userns: No such file or directory
/proc/sys/kernel/core_pattern = |/wsl-capture-crash %t %E %p %s
/proc/sys/kernel/core_pipe_limit = 0
/proc/sys/kernel/yama/ptrace_scope = 1
/proc/sys/kernel/unprivileged_bpf_disabled = 2
/proc/sys/vm/unprivileged_userfaultfd = 0
cat: /sys/kernel/security/lsm: No such file or directory
ulimit -c soft=0 hard=unlimited
CORE     max core file size    0 unlimited bytes
CONFIG_CGROUPS=y  CONFIG_MEMCG=y  CONFIG_USER_NS=y  CONFIG_IO_URING=y  CONFIG_SECCOMP_FILTER=y  CONFIG_OVERLAY_FS=y

$ wsl.exe ... 'ps -p 1 -o pid,comm,args; systemctl is-system-running; systemctl --user is-system-running; echo XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR DBUS_SESSION_BUS_ADDRESS=$DBUS_SESSION_BUS_ADDRESS; stat -fc %T /sys/fs/cgroup; cat /sys/fs/cgroup/cgroup.controllers; cat /proc/self/cgroup; [user@1000.service cgroup.controllers and subtree_control]; systemctl show user@1000.service -p Delegate -p DelegateControllers'
      1 systemd         /sbin/init
running
running
XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
cgroup2fs
cpuset cpu io memory hugetlb pids rdma
0::/non-systemd
user@ controllers: cpu memory pids
user@ subtree_control: cpu memory pids
Delegate=yes
DelegateControllers=cpu memory pids

$ wsl.exe ... 'unshare -r true; echo $?; unshare -rinmpfu --mount-proc --kill-child true; echo $?; unshare -rinmpfu --mount-proc --kill-child sh -c "id -u; cat /proc/self/uid_map; echo pid=\$\$; ls /proc | grep -c ^[0-9]; cat /proc/sys/user/max_user_namespaces; echo 0 > /proc/sys/user/max_user_namespaces && echo wrote0=\$(cat /proc/sys/user/max_user_namespaces); ip link | head -2"'
unshare -r: 0
unshare -rinmpfu --mount-proc --kill-child: 0
0
         0       1000          1
pid=1
3
2147483647
wrote0=0
1: lo: <LOOPBACK> mtu 65536 qdisc noop state DOWN mode DEFAULT group default qlen 1000

$ wsl.exe ... 'systemd-run --user --scope --quiet -p MemoryMax=64M -p MemorySwapMax=0 -p TasksMax=64 -p RuntimeMaxSec=20 -p TimeoutStopSec=1 sh -c "cat /proc/self/cgroup; d=/sys/fs/cgroup\$(cut -d: -f3 /proc/self/cgroup); echo memory.max=\$(cat \$d/memory.max) memory.swap.max=\$(cat \$d/memory.swap.max) pids.max=\$(cat \$d/pids.max)"'
0::/user.slice/user-1000.slice/user@1000.service/app.slice/run-rbaeccf27fb0040aca9d3ade54d946d6c.scope
memory.max=67108864 memory.swap.max=0 pids.max=64
systemd-run rc=0
```

[MEASURED 2026-10-15, workstation]: every tool the remote tests check is on PATH, and python3 is in SANDBOX_PATH. systemd is PID 1, and the user manager runs with its bus at /run/user/1000/bus. Unprivileged user namespaces work, and so does the sandbox's own `unshare -rinmpfu --mount-proc --kill-child`: uid 0 inside maps to 1000, the shell is pid 1 with only 3 processes in view, only lo exists and is down, and the namespace's max_user_namespaces accepts the write of 0. No AppArmor userns restriction sysctl exists, and no LSM list is exposed. `systemd-run --user --scope` applies MemoryMax, MemorySwapMax=0, and TasksMax (read back as 67108864, 0, and 64). The user manager delegates cpu, memory, and pids, so the cpu controller is delegated here, unlike alpha01 (OQ-011). Processes that `wsl.exe --exec` starts sit in cgroup /non-systemd, outside any session, yet systemd-run --user still placed its command in a user scope.

### The setup's mounts (the exit 16 in plans/LESSONS.md)

The setup script was taken from the clean tree at 422876c, without running it on the host or writing bytecode:

```
$ .venv/Scripts/python.exe -B -c "from lassi.executors import sandbox as s; open('<scratch>/setup.sh','w',newline='\n').write(s.SETUP_SCRIPT); ..."
SETUP_SCRIPT bytes 10901 sha256 3ec98d452e7677a8
SANDBOX_PATH /usr/sbin:/usr/bin:/sbin:/bin
```

The probe script (session scratch, mountprobe.sh; run as `wsl.exe -d Ubuntu-24.04 --exec bash -lc 'cd / && sh <scratch>/mountprobe.sh <scratch>'`). Every mount happens in a private user and mount namespace. An outer `unshare -rm` puts a tmpfs on /srv to hold the workdir, so the copy-back writes nowhere on the host. That nesting is the one difference from a production run, which starts the sandbox's unshare from the host user directly.

```
#!/bin/sh
# [comment lines trimmed]
d=$1
echo "== M1 minimal: tmpfs, then mount --move, in unshare -rm (no read-only step)"
unshare -rm sh -c '
mount -t tmpfs probe-stage /tmp
mkdir /tmp/a /tmp/b /tmp/c /tmp/e
mount -t tmpfs probe-x /tmp/a
mount --move /tmp/a /tmp/b; echo "plain mount --move rc=$?"
grep -c " /tmp/b " /proc/self/mountinfo | sed "s/^/  mountinfo lines at \/tmp\/b: /"
mount -t tmpfs probe-y /tmp/c
mount -n --move /tmp/c /tmp/e; echo "mount -n --move rc=$?"
grep -c " /tmp/e " /proc/self/mountinfo | sed "s/^/  mountinfo lines at \/tmp\/e: /"
mount -t tmpfs probe-z /tmp/c
mount -o remount,bind,ro /tmp/c; echo "plain remount,bind,ro rc=$?"
mount --bind /tmp/c /tmp/a; echo "plain --bind rc=$?"
' 2>&1
run_setup() {
  echo "== $1"
  unshare -rm sh -c '
  mount -t tmpfs probe-outer /srv && mkdir /srv/wd && cd /
  prlimit --core=1 -- env -i PATH=/usr/sbin:/usr/bin:/sbin:/bin \
    unshare -rinmpfu --mount-proc --kill-child sh -c "$(cat "$1")" sh /srv/wd "" "" 8388608 1 /home/joe 10 5 2 \
    sh -c "echo program-ran; id -u; ls /home/joe | wc -l; echo hi > out.txt"
  echo "setup rc=$?"
  echo "workdir after: $(ls /srv/wd)"
  ' sh "$2" 2>&1 | tail -n 12
}
run_setup "S1 SETUP_SCRIPT as committed" "$d/setup.sh"
sed -e 's/^\( *\)mount /\1mount -n /' "$d/setup.sh" > "$d/setup-n.sh"
echo "(S2 variant: $(grep -c "mount -n " "$d/setup-n.sh") mount lines given -n, umount lines unchanged)"
run_setup "S2 SETUP_SCRIPT with mount -n on every mount line (exploratory variant, not committed)" "$d/setup-n.sh"
sed -e 's/^\( *\)mount --move /\1mount -n --move /' "$d/setup.sh" > "$d/setup-move.sh"
echo "(S3 variant: $(grep -c "mount -n --move " "$d/setup-move.sh") mount --move lines given -n, nothing else changed)"
run_setup "S3 SETUP_SCRIPT with -n on the --move lines only (exploratory variant, not committed)" "$d/setup-move.sh"
echo "== S4 full composition (prlimit, env -i, systemd-run --user --scope with limits, unshare) with the S3 variant"
unshare -rm sh -c '
mount -t tmpfs probe-outer /srv && mkdir /srv/wd && cd /
prlimit --core=1 -- env -i PATH=/usr/sbin:/usr/bin:/sbin:/bin XDG_RUNTIME_DIR=/run/user/1000 \
  DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus \
  systemd-run --user --scope --quiet -p MemoryMax=512M -p MemorySwapMax=0 -p TasksMax=64 -p RuntimeMaxSec=22 \
    -p TimeoutStopSec=1 \
  unshare -rinmpfu --mount-proc --kill-child sh -c "$(cat "$1")" sh /srv/wd "" "" 8388608 1 /home/joe 10 5 2 \
    sh -c "echo program-ran; cat /proc/self/cgroup; echo hi > out.txt"
echo "setup rc=$?"
echo "workdir after: $(ls /srv/wd)"
' sh "$d/setup-move.sh" 2>&1 | tail -n 12
```

A first run passed a hidden-root count of 0 and stopped with status 2 at the script's own number check (`'' | 0* | *[!0-9]*) exit 2`, sandbox.py:694). The sandbox never sends 0, since SandboxSpec refuses a spec with no hidden root (sandbox.py:1132). The script above passes one hidden root, /home/joe. The script ran three times: first with the count of 0, then as M1, S1, and S2 with one hidden root, and last with S3 and S4 appended, its output filtered to them. Below are M1 to S2 from the second run and S3 and S4 from the third (the count lines "22 mount lines given -n" and "4 mount --move lines given -n" trimmed):

```
== M1 minimal: tmpfs, then mount --move, in unshare -rm (no read-only step)
mount: /tmp/b: filesystem was mounted, but failed to update userspace mount table.
plain mount --move rc=16
  mountinfo lines at /tmp/b: 1
mount -n --move rc=0
  mountinfo lines at /tmp/e: 1
plain remount,bind,ro rc=0
plain --bind rc=0
== S1 SETUP_SCRIPT as committed
mount: /dev: filesystem was mounted, but failed to update userspace mount table.
setup rc=16
workdir after:
== S2 SETUP_SCRIPT with mount -n on every mount line (exploratory variant, not committed)
lassi-sandbox-ready
program-ran
0
0
lassi-sandbox-done 3288.65 3288.67
setup rc=0
workdir after: out.txt
== S3 SETUP_SCRIPT with -n on the --move lines only (exploratory variant, not committed)
lassi-sandbox-ready
program-ran
0
0
lassi-sandbox-done 3308.37 3308.39
setup rc=0
workdir after: out.txt
== S4 full composition (prlimit, env -i, systemd-run --user --scope with limits, unshare) with the S3 variant
lassi-sandbox-ready
program-ran
0::/user.slice/user-1000.slice/user@1000.service/app.slice/run-r23252406fd814b94af9a860f66a6e6e3.scope
lassi-sandbox-done 3308.54 3308.56
setup rc=0
workdir after: out.txt
```

The four lines S3 changed are sandbox.py:728 (`mount --move /tmp/dev /dev`), :742 (workdir), :744 (harness), and :747 (toolchains).

[MEASURED 2026-10-15, workstation, script text from commit 422876c, util-linux 2.39.3, kernel 6.18.54.1]: in a user namespace, util-linux 2.39.3's `mount --move` moves the mount (mountinfo shows it at the target) and then exits 16 with "filesystem was mounted, but failed to update userspace mount table". `mount -n --move` exits 0. Plain `--bind` and `remount,bind,ro` exit 0. The committed SETUP_SCRIPT stops at its first move, `mount --move /tmp/dev /dev` (sandbox.py:728), with status 16, before the ready marker, so the program never runs. The sandbox reports such a run as SandboxUnavailableError (its docstring). The mount(8) page shipped in WSL explains the status: exit 16 is "problems writing or locking /etc/mtab" (mount.8.gz lines 2447-2449), and -n is "Mount without writing in /etc/mtab" (lines 422-424). libmount keeps userspace options in /run/mount/utab (doc, libmount "Tables update"); /run/mount there is root-owned (`drwxr-xr-x root root`; section "Probes of the planning round (2026-10-15)"), which the namespace cannot map, so the utab update failing there is the likely cause (inferred, not traced).

(exploratory) With `-n` added to the four `mount --move` lines only (S3), the committed setup ran end to end in WSL. That means ready marker, program as uid 0 of its namespace, /home/joe hidden (0 entries), done marker, and the copy-back of out.txt. S4 adds the production wrappers: prlimit --core=1, env -i with the bus variables, and systemd-run --user --scope with memory, swap, task, and runtime limits. It ran the same way, with the program inside the transient scope. S2 (-n on all 22 mount lines) also passed, so the smallest change is the four move lines. Whether alpha01's util-linux 2.37.2 accepts `-n --move` the same way is not measured; alpha01 is out of reach. -n is a long-standing mount option (inferred), and PHASE-NOTES (P17 candidates) asks for exactly this measurement before the edit.

### Core handling

```
$ wsl.exe ... 'ls -l /wsl-capture-crash; systemctl is-active systemd-coredump.socket apport; ls /var/lib/systemd/coredump | head -3; ls /var/crash | head -3'
ls: cannot access '/wsl-capture-crash': No such file or directory
inactive
inactive
[both directories listed empty]
$ ls -la /c/Users/Joseph/AppData/Local/Temp/wsl-crashes | head -5
-rw-r--r-- 1 Joseph 197121 188063744 Oct 14 23:04 wsl-crash-1792033446-608-_home_joe_partytrix-gate_tor_tor-11.dmp
```

[MEASURED 2026-10-15, workstation]: the kernel pipes cores to `|/wsl-capture-crash %t %E %p %s`. That handler is not visible inside the distro (it belongs to the WSL VM, inferred), and it stores dumps on Windows under %TEMP%\wsl-crashes on C: (doc: crashDumpFolder default; maxCrashDumpCount default 10). One 188 MB dump of another project's process from 2026-10-14 is there. A login shell's core limit is soft 0 and hard unlimited, and neither systemd-coredump nor apport is active. The sandbox sets `prlimit --core=1` for the whole run, and the program again, which on alpha01 made the kernel abort a piped core (plans/spikes/p0-sandbox-hardening.md, probes F and G). That this also holds on WSL kernel 6.18 is not measured here, since a crash test would write to the Windows dump folder if it did not hold (inferred from the kernel's limit-1 rule for piped cores).

## 5. The repository's Linux-only tests in WSL today

```
$ wsl.exe ... 'ls -la ~ | head -40; du -sh ~/* ~/.cache ~/.local | sort -h | tail -15; find ~ -maxdepth 3 -name pyvenv.cfg; find ~ -maxdepth 3 -name .git -type d'
[... trimmed to the relevant entries ...]
drwxr-xr-x  5 joe  joe  4096 Oct 11 00:18 .lassi-audit-venv
drwxr-xr-x  5 joe  joe  4096 Oct 11 00:18 lassi-audit
11M	/home/joe/lassi-audit
8.1G	/home/joe/mlir_dev
/home/joe/.lassi-audit-venv/pyvenv.cfg
/home/joe/temp_venv/pyvenv.cfg
/home/joe/xdsl/.git
/home/joe/mlir_dev/llvm-project/.git
$ wsl.exe ... 'ls -la ~/lassi-audit; cat ~/.lassi-audit-venv/pyvenv.cfg; ~/.lassi-audit-venv/bin/python -m pip list'
assets  lassi  pyproject.toml  tests            [no .git, no uv.lock, no tools/]
version = 3.12.3
executable = /usr/bin/python3.12
pip 24.0, pytest 9.1.1, PyYAML 6.0.3, pyarrow 26.0.0, tiktoken 0.14.0, regex 2026.9.29, requests 2.34.2, Pygments, iniconfig, packaging, pluggy, certifi, charset-normalizer, idna, urllib3
$ wsl.exe ... 'diff -rq -x __pycache__ ~/lassi-audit/lassi /mnt/c/dev/lassi/lassi; diff -rq -x __pycache__ ~/lassi-audit/tests /mnt/c/dev/lassi/tests; diff -q .../executors/sandbox.py && echo sandbox.py-identical'
Files .../lassi/corpus/record.py differ
Files .../lassi/corpus/tables.py differ
Files .../lassi/ir/form.py differ
Files .../tests/ir/test_ir_form.py differ
Files .../tests/toolchains/test_pins.py differ
Only in /mnt/c/dev/lassi/tests/toolchains: test_tt_mlir_pins.py
Only in /mnt/c/dev/lassi/tests/toolchains: test_tt_mlir_remote.py
Files .../tests/toolchains/test_tt_pins.py differ
Files .../tests/tools/test_status.py differ
sandbox.py-identical
[its __pycache__ holds cpython-312-pytest-9.1.1 files from 2026-10-11 00:18]
```

[MEASURED 2026-10-15, workstation]: WSL holds no git checkout of the repository. ~/lassi-audit is a partial file copy (assets, lassi, tests, pyproject.toml) from 2026-10-11 00:18 that differs from HEAD in 9 entries. Its sandbox.py matches HEAD. Its venv, ~/.lassi-audit-venv, runs Python 3.12.3, outside the project's requires-python (>=3.10,<3.11), and has neither the lock nor the cpu extra (no numpy or torch). Pytest last ran there on 2026-10-11 (its bytecode), presumably for the P5.11 audit (inferred from the date). WSL has no uv and no Python 3.10.

So the Linux-only tests do not run in WSL at HEAD today; nothing there can run them as the project does (uv, Python 3.10, the lock). On Windows they skip. By source, the local tests that skip off Linux or POSIX are these: tests/toolchains/test_runner_caps.py:265 (ru_maxrss), tests/executors/test_sandbox.py:1694 (POSIX permission bits) and :1502 (unshare -r), tests/executors/test_ttsim_findings.py:297, tests/present/test_present_settings.py:287, tests/tools/test_gate_windows.py:34 (posix_only), and the symbolic-link tests where links cannot be made. CI runs the device-neutral list on ubuntu-latest (.github/workflows/cpu-tests.yml:16, `-m "not remote"`), and the local sandbox tests there use stub mount tools, not the real setup (inferred from tests/executors/test_sandbox.py's stubs). The remote-marked sandbox tests would find every prerequisite host_problem checks in WSL once LASSI_SCRATCH is set. They would then fail, not skip, at the setup's exit 16 (inferred from S1 and tests/executors/test_sandbox_remote.py:178-194; not run).

## Probes of the planning round (2026-10-15)

- Date: 2026-10-15, 01:09 to 01:11 EDT, on the workstation. Base: branch p5-ir at 422876c, with this spike and plans/local-plan.md untracked.
- Why: a review of plans/local-plan.md found values given as [MEASURED] whose commands and outputs were not logged, here or in the plan. Each was re-run read-only, and is logged below.
- Remote and devices: none, as above. The GPU probes list paths only and open no device.
- Writes: none. Ubuntu-24.04 was stopped at 01:09, so the first WSL probe started it.
- Shell: as above. WSL probes ran as `MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 --exec bash -lc '<cmd>'` and are shown as `wsl ... '<cmd>'`. Outputs have CR bytes stripped.

```
$ date -Is; wsl.exe -l -v
2026-10-15T01:09:36-04:00
  NAME              STATE           VERSION
* Ubuntu-24.04      Stopped         2
  docker-desktop    Stopped         2

$ powershell.exe -NoProfile -Command "Get-Date -Format o; (Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToString('o')"
2026-10-15T01:09:58.5969261-04:00
2026-10-14T13:44:42.5000000-04:00

$ wsl ... 'date -Is; uptime -s; cat /proc/uptime; grep -E "^(MemTotal|SwapTotal):" /proc/meminfo; nproc'
2026-10-15T01:10:04-04:00
2026-10-15 01:10:00
4.16 47.28
MemTotal:       16334740 kB
SwapTotal:       4194304 kB
16

$ wsl ... 'dpkg-query -W -f="\${Package} \${db:Status-Abbrev} \${Version}\n" libhwloc-dev libnuma-dev libtbb-dev libcapstone-dev libssl-dev pkg-config pkgconf xz-utils; echo "dpkg-query rc=$?"; command -v pkg-config || echo "pkg-config: not on PATH"'
dpkg-query: no packages found matching libhwloc-dev
dpkg-query: no packages found matching libnuma-dev
dpkg-query: no packages found matching libtbb-dev
dpkg-query: no packages found matching libcapstone-dev
dpkg-query: no packages found matching libssl-dev
dpkg-query: no packages found matching pkg-config
dpkg-query: no packages found matching pkgconf
xz-utils ii  5.6.1+really5.4.5-1ubuntu0.3
dpkg-query rc=1
pkg-config: not on PATH

$ wsl ... 'for t in rustc cargo rustup dotnet flang flang-new flang-new-17 flang-new-20 gfortran clang apptainer singularity; do p=$(command -v $t 2>/dev/null); printf "%-14s %s\n" "$t" "${p:-MISSING}"; done; dpkg-query -W -f="\${Package} \${db:Status-Abbrev} \${Version}\n" "rustc*" "cargo*" "dotnet*" "flang*" "gfortran*" "clang*" "apptainer*" "singularity*" 2>&1; echo "dpkg-query rc=$?"'
rustc          MISSING
cargo          MISSING
rustup         MISSING
dotnet         MISSING
flang          MISSING
flang-new      MISSING
flang-new-17   MISSING
flang-new-20   MISSING
gfortran       MISSING
clang          MISSING
apptainer      MISSING
singularity    MISSING
dpkg-query: no packages found matching rustc*
dpkg-query: no packages found matching cargo*
dpkg-query: no packages found matching dotnet*
dpkg-query: no packages found matching flang*
dpkg-query: no packages found matching gfortran*
dpkg-query: no packages found matching clang*
dpkg-query: no packages found matching apptainer*
dpkg-query: no packages found matching singularity*
dpkg-query rc=1

$ wsl ... 'ls -ld /dev/dxg /usr/lib/wsl/lib 2>&1; ls /usr/lib/wsl/lib 2>&1 | head -20; ls -d /dev/nvidia* 2>&1; ls -d /dev/nvidiactl /proc/driver/nvidia /proc/driver/nvidia/gpus 2>&1; ls -ld /run/mount 2>&1; stat -c "%A %U %G %n" /run/mount /run/mount/utab 2>&1'
crw-rw-rw- 1 root root 10, 258 Oct 15 01:10 /dev/dxg
drwxr-xr-x 1 root root      80 Oct 15 01:10 /usr/lib/wsl/lib
libcuda.so
libcuda.so.1
libcuda.so.1.1
libcudadebugger.so.1
libd3d12.so
libd3d12core.so
libdxcore.so
libnvcuvid.so
libnvcuvid.so.1
libnvdxdlkernels.so
libnvidia-encode.so
libnvidia-encode.so.1
libnvidia-gpucomp.so
libnvidia-gpucomp.so.615.78.05
libnvidia-ml.so.1
libnvidia-ngx.so.1
libnvidia-opticalflow.so
libnvidia-opticalflow.so.1
libnvoptix.so.1
libnvoptix_loader.so.1
ls: cannot access '/dev/nvidia*': No such file or directory
ls: cannot access '/dev/nvidiactl': No such file or directory
ls: cannot access '/proc/driver/nvidia': No such file or directory
ls: cannot access '/proc/driver/nvidia/gpus': No such file or directory
drwxr-xr-x 2 root root 40 Oct 15 01:10 /run/mount
drwxr-xr-x root root /run/mount
stat: cannot statx '/run/mount/utab': No such file or directory

$ wsl ... 'ls /usr/lib/wsl/lib | wc -l; ls /usr/lib/wsl/lib | tail -n +21; uname -r; date -Is'
23
libnvwgf2umx.so
nvidia-smi
6.18.54.1-microsoft-standard-WSL2
2026-10-15T01:10:24-04:00
```

[MEASURED 2026-10-15 01:09 to 01:11, workstation, WSL kernel 6.18.54.1]:

- Windows last booted at 2026-10-14 13:44:42 EDT, so at 00:23 it had been up 10 h 38 min. WSL's VM started at 01:10:00 for these probes, so its uptime in the first round cannot be re-read and is not given.
- After the restart, MemTotal read 16,334,740 kB (15,951.9 MiB; `free -m` in section 2 printed 15951), SwapTotal 4,194,304 kB (4 GiB), and nproc 16. The first round's MemTotal figure, 16,334,724 kB, was never logged, and this round reads 16 kB more. That figure is dropped.
- tt-metal's build packages: dpkg knows none of libhwloc-dev, libnuma-dev, libtbb-dev, libcapstone-dev, libssl-dev, pkg-config, or pkgconf, and pkg-config is not on PATH. xz-utils is installed (`ii`, 5.6.1+really5.4.5-1ubuntu0.3).
- P12's and P13's tools: rustc, cargo, rustup, dotnet, flang (as flang, flang-new, flang-new-17, and flang-new-20), gfortran, clang, apptainer, and singularity are not on the login shell's PATH. dpkg knows no package matching rustc*, cargo*, dotnet*, flang*, gfortran*, clang*, apptainer*, or singularity*.
- GPU stack: /dev/dxg exists (a character device, 10, 258, mode crw-rw-rw-). /usr/lib/wsl/lib holds 23 entries, among them libcuda.so.1, libnvidia-ml.so.1, and nvidia-smi. No /dev/nvidia* node exists, and neither does /dev/nvidiactl or /proc/driver/nvidia, so /proc/driver/nvidia/gpus is absent too.
- /run/mount is `drwxr-xr-x root root`. It held no utab at the probe, just after the restart had cleared /run.

## Sources

- Microsoft Learn, "Advanced settings configuration in WSL", https://learn.microsoft.com/en-us/windows/wsl/wsl-config (page updated_at 2026-09-16, read 2026-10-15): the .wslconfig defaults for memory (50% of Windows memory), processors (all), swap (25% of memory rounded up to the nearest GB), maxCrashDumpCount (10), and crashDumpFolder (%Temp%\wsl-crashes).
- util-linux 2.39.3's mount(8), /usr/share/man/man8/mount.8.gz in WSL Ubuntu 24.04: lines 422-424 (-n, --no-mtab) and 2447-2449 (exit status 16).
- libmount "Tables update" documentation (utab under /run/mount), https://www.kernel.org/pub/linux/utils/util-linux/v2.24/libmount-docs/libmount-Tables-update.html (read 2026-10-15).
- Repository at 422876c: lassi/executors/sandbox.py (docstring; :682-800 the setup template; :694, :728, :742, :744, :747, :1132), lassi/executors/gpu.py:104, tests/executors/test_sandbox_remote.py:126 and :178-194, toolchains/*.pin and *.sh as cited, .github/workflows/cpu-tests.yml, plans/spikes/p0-sandbox-hardening.md (probes F and G), plans/PHASE-NOTES.md (P17 candidates, the util-linux 2.39 note), plans/LESSONS.md.

## Finding

The workstation has the raw capacity for LASSI's local work. Measured: an i7-10700K (8 cores, 16 threads), 31.9 GiB of RAM, of which WSL gets the defaults (16 CPUs, 15.6 GiB, 4 GiB swap, since .wslconfig is empty), and 721 GiB free on the NVMe drive that holds the WSL disk. WSL Ubuntu 24.04 also has everything the sandbox's mechanism needs (user namespaces, the systemd user manager with cpu, memory, and pids delegated, overlay, seccomp). Two gaps stop local trials and toolchain builds. First, the committed sandbox setup exits 16 at `mount --move /tmp/dev /dev` under WSL's util-linux 2.39.3; adding -n to its four `mount --move` lines made the full composition run (exploratory). Second, none of the pinned host tools is installed: WSL has gcc 13.3.0, cmake 3.28.3, ninja 1.11.1, and Python 3.12.3, with no clang, lld, uv, or Python 3.10, while the pins name alpha01's Ubuntu 22.04 tools, and every install script refuses any root outside /mnt/nvme10. The workstation also holds an NVIDIA RTX 3080 (10 GiB, compute capability 8.6), visible in WSL through /dev/dxg, not through the /dev/nvidia* nodes the gpu executor binds.

Confidence: high for the inventory, the tool versions, the namespace and scope probes, and the exit 16 (each reproduced in this session). Medium for the -n fix: it was measured on WSL only, with the outer namespace nesting described above, not on alpha01's util-linux 2.37.2. Low for any build-time or memory projection from the old LLVM log, since the log does not record its -j or why it restarted.

## Consequences for the plan

1. Sandbox (a code task, [DESIGN], with a Decision Log entry): add -n to the four `mount --move` lines of the setup template (sandbox.py:728, :742, :744, :747), with a test that pins it. Then run tests/executors/test_sandbox_remote.py in WSL with LASSI_REQUIRE_SANDBOX=1 from a clean checkout. alpha01's 2.37.2 stays unmeasured until it is reachable again. sandbox.py's docstring says the composition counts as proven only once those remote tests pass from a clean commit. A WSL pass would be the first on this host and must name it as the device, not alpha01.
2. Linux-only tests: they need a WSL checkout at HEAD with uv and Python 3.10 (a user-level install; the owner decides). Until then a Windows run plus CI is the coverage, and CI exercises no real sandbox.
3. P5.4 to P5.G (tt-mlir, Polygeist, tt-metal, ttsim): as committed, nothing installs here, since the scripts refuse the root and the host tools. Building locally needs an owner decision on where (draft A below): disk is ample (tt-mlir.pin INSTALL_KIB plus RESERVE_KIB is about 42.5 GiB against 721 GiB free on C:), and memory is the tight resource for LLVM-scale builds at 16 jobs in 15.6 GiB (PROJECTED, from the log's pattern; not measured).
4. Pins: a toolchain built from WSL Ubuntu 24.04's packages cannot match the current pins' EXPECT strings, and changing a pin needs the owner (Agent Rule 10). An Ubuntu 22.04 WSL distro would likely match more of them: jammy ships Python 3.10.12, and ninja 1.10.1 is the pinned version. Since this spike, plans/local-plan.md section 2.3 found the exact clang-20 build the pin names (1~exp1~20250708202428.132) still served by apt.llvm.org for jammy (index, read 2026-10-15; nothing installed).
5. GPU: the RTX 3080 is the first NVIDIA device within LASSI's reach. It is not an A100 (P10) and not a ROCm card (P9). In WSL its device is /dev/dxg plus /usr/lib/wsl/lib, so the gpu executor's NVIDIA exposure (gpu.py:104: /dev/nvidiactl, /dev/nvidia-uvm, /dev/nvidia-uvm-tools, /dev/nvidia<N>) would not find its nodes there. Whether it counts as a GPU host for P17's part (b) is the owner's call (draft B).
6. Results made here name the workstation, its WSL kernel, and its util-linux as the device and environment. ttsim results stay simulator results (Agent Rule 2).

## Proposed bible edit (factual) and Decision Log entry

plans/local-plan.md section 6.1 carries these drafts forward as factual edits, through the bible-sync procedure.

Environment State, a new table row after Storage:

| Workstation | The only host while hardware and alpha01 are unavailable (owner, 2026-10-15). Windows 10 Home 22H2 (build 19045.7725); Intel Core i7-10700K, 8 cores and 16 threads; 31.9 GiB RAM; NVIDIA GeForce RTX 3080, 10 GiB, compute capability 8.6, driver 617.42. WSL 3.0.3.0, kernel 6.18.54.1, Ubuntu 24.04.2: 16 CPUs, 15.6 GiB memory and 4 GiB swap (the WSL defaults; .wslconfig sets nothing), systemd user manager with cpu, memory, and pids delegated, unprivileged user namespaces, util-linux 2.39.3, gcc 13.3.0, cmake 3.28.3, ninja 1.11.1, Python 3.12.3, no clang, lld, uv, or Python 3.10. Free space: C: 721 GiB (holds the WSL disk), D: 387, E: 314, F: 1648 (USB), Z: 578 [MEASURED 2026-10-15, workstation] | plans/spikes/local-host.md (CIM queries, `wsl -l -v`, `nproc`, `free`, `df`, tool versions, namespace and scope probes) | The sandbox's setup stops at `mount --move` there (Sandbox); no pinned host tool is installed, and the install scripts refuse roots outside /mnt/nvme10; where to build is OQ-<next> |

Environment State, the NVIDIA GPU row's state cell gains: "J's workstation holds a GeForce RTX 3080 (10 GiB, compute capability 8.6), seen in WSL through /dev/dxg with no /dev/nvidia* node [MEASURED 2026-10-15, workstation, plans/spikes/local-host.md]; whether it serves as a GPU host is OQ-<next+1>."

Sandbox, after the mechanism bullet:

"- util-linux 2.39 [MEASURED 2026-10-15, workstation: WSL Ubuntu 24.04, util-linux 2.39.3, kernel 6.18.54.1, script text from commit 422876c; plans/spikes/local-host.md]: in the sandbox's user namespace `mount --move` moves the mount and then exits 16 ("failed to update userspace mount table"), so the committed setup stops at `mount --move /tmp/dev /dev` before the ready marker and the program never runs; `mount -n --move` exits 0. With -n on the setup's four `mount --move` lines, the whole composition ran there (exploratory, an uncommitted variant). The probe nested the sandbox's unshare inside an outer `unshare -rm` that held the workdir on a tmpfs, the one difference from a production run, which starts the sandbox's unshare from the host user directly. sandbox.py has changed since P0.16 (1de7db6, 6566e87, b9fc0d3, and bf98a81), and bf98a81 is its last change before 422876c. On alpha01, util-linux 2.37.2 ran that setup: the sandbox's remote tests passed at bf98a81 (P17.5, rx 20261006-082332-desktop-8r113ei-detached-bf98a813-c6b6, clean), and later clean runs went through the same setup (P17.6, rx 20261006-121638-desktop-8r113ei-detached-f03fa62a-961a; P5.3, rx 20261010-184351-p53-tier-b-e64d)."

Decision Log, new top row (and the count sentence becomes "One hundred and forty-one decisions ... and one on 2026-10-12, and one on 2026-10-15"):

| 2026-10-15 | The workstation's capacity is recorded (Environment State, the Workstation row and the NVIDIA GPU row; Sandbox, the util-linux 2.39 bullet; plans/spikes/local-host.md) | The owner's direction of 2026-10-15 makes the workstation the only host. Read-only local probes measured it, and the committed sandbox setup was found to stop at `mount --move` under WSL's util-linux 2.39.3. The code fix is a separate task with its own entry |

## Owner-queue drafts (choices)

Superseded by plans/local-plan.md section 7: draft A by its decision 1, draft B by its decision 8. Neither draft is to be filed as written. Their stale statements are corrected below, so they no longer contradict the plan:
- Draft A's option (b) no longer amends Agent Rule 7. The plan's decision 3 recommends writing the workstation's policy in Host Facts and an AGENTS.md Local Execution section, with the rules unchanged.
- The pinned clang-20 build is no longer unverified. apt.llvm.org still serves it for jammy (plans/local-plan.md section 2.3; index, read 2026-10-15).

Draft A:

```
## OQ-<next> Where LASSI Builds And Runs While Only The Workstation Exists
State: OPEN
Kind: decision
Blocks: P5.4, P5.5, P5.6, P5.7, P5.8, P5.9, P5.10, P5.12, P5.G
Evidence: plans/spikes/local-host.md
Question: With alpha01 and every accelerator unavailable (owner, 2026-10-15), P5's toolchains (tt-mlir, Polygeist, tt-metal, ttsim) and every sandboxed compile and run need a Linux host. The workstation's WSL Ubuntu 24.04 has the capacity (16 CPUs, 15.6 GiB, 721 GiB free on C:) but none of the pinned host tools (gcc 13.3.0, cmake 3.28.3, ninja 1.11.1, Python 3.12.3; no clang, lld, uv, or Python 3.10), and every install script refuses roots outside /mnt/nvme10. Where and how should LASSI build locally?
Options:
(a) Wait for alpha01. Nothing is installed. P5 does only Windows halves after P5.11, and local work is pure Python, the Windows suite, and the sandbox -n fix tested in WSL.
(b) A dedicated Ubuntu 22.04 WSL distro that mirrors alpha01's tools (g++-12, Python 3.10, ninja 1.10.1, CMake 3.31.6 under /usr/local, clang-17 and clang-20 with lld from apt.llvm.org, uv). The install scripts take their root and stop line from explicit host-profile values with no default, the workstation's policy goes in Host Facts and an AGENTS.md Local Execution section with Agent Rule 7 unchanged, and .wslconfig memory may be raised (about 24 GiB, a proposal) for LLVM builds. The pins stay as they are, since the package indexes still serve every pinned host-tool build (plans/local-plan.md section 2.3), so results stay comparable with alpha01's P4 and P5 evidence. It needs the owner to install the distro and packages (or approve an agent doing so) and to change .wslconfig. apt.llvm.org may later replace the pinned clang-20 build, so the fetched packages should be kept.
(c) Re-pin to the existing Ubuntu 24.04 distro's tools plus what it lacks. This is quicker to set up but changes several pins (Agent Rule 10, Decision Log), and evidence built under the old and new pins needs care when compared. It also shares the distro with the owner's other projects.
Recommendation: (b), because it keeps the pins and the evidence comparable, and it keeps LASSI's toolchains apart from the other projects in Ubuntu-24.04. The sandbox fix (-n on the four `mount --move` lines) is a normal task under every option.
Answer:
```

Draft B:

```
## OQ-<next+1> Is The Workstation's RTX 3080 A GPU Host For P17's GPU Half?
State: OPEN
Kind: decision
Blocks: P17.12, P17.13, P17.G
Evidence: plans/spikes/local-host.md
Question: The workstation holds an NVIDIA GeForce RTX 3080 (10 GiB, compute capability 8.6, driver 617.42), visible in WSL through /dev/dxg and /usr/lib/wsl/lib. P17's GPU half waits for a GPU host (OQ-040), and the bible says J has no NVIDIA host (OQ-003). The gpu executor binds the /dev/nvidia* nodes of a Linux host, which WSL does not have. Should the RTX 3080 serve as a GPU host?
Options:
(a) No. P17.12, P17.13, and P17.G stay BLOCKED as now.
(b) For P17.12's spike only: measure what CUDA needs inside the sandbox in WSL (/dev/dxg, /usr/lib/wsl/lib), with no model inference or training, then decide again. It uses the owner's display GPU briefly, and any device exposure for WSL becomes a design change to the gpu executor.
(c) For all of P17's part (b): the gpu executor, the nvml profiler, hf_local on cuda, and short training runs. Models must fit in 10 GiB, and it is a consumer card, not the A100 of P10 or the MI300X of P8 and P9, so P10's reproduction stays blocked.
Recommendation: (b), because the WSL GPU path differs from the Linux one the executor was designed for, and a spike settles what it needs before any run depends on it.
Answer:
```
