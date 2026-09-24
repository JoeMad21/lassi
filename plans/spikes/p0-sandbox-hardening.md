# Spike P0.16: unprivileged mechanisms for sandbox hardening on alpha01

- Task: P0.16 (plans/p0-core.md). Bible: Sandbox; Agent Rules 5, 6, 7, 9, 12. Earlier evidence: plans/spikes/p0-sandbox.md, plans/spikes/p0-sandbox-verify.md, and the Known limits in the lassi/executors/sandbox.py docstring.
- Date: probes A to K ran on 2026-09-23 between 13:26 and 13:45 (alpha01 clock, UTC-07:00). Each rx id below carries its own start time. Probe L, in the fixer addendum, ran later that day on another base; its date, base, and status are in the addendum.
- Local commit when probes A to K ran: 683b975 (branch p0-core). The tree was dirty only in plans/STATUS.md. Probes A to K used `uv run tools/rx.py exec`, which runs from the scratch root without a checkout, with inline scripts only, so no repository code took part. Probe L embedded script text from an uncommitted tree, so it is exploratory.
- Host: alpha01. Kernel 6.6.29+main+3.0.0r1-amd64-gio-epilmore-dev+, util-linux 2.37.2 (unshare, mount), GNU coreutils 8.32 (timeout), Python 3.10.12 (/usr/bin/python3), systemd 249 (from plans/spikes/p0-sandbox.md; not re-read here). Device: host CPU and kernel only. No device command was run, no accelerator node was opened, and no probe command named an accelerator node.
- Access: `rx exec` only, one command at a time. Each probe that wrote files made its own `mktemp -d -p "$LASSI_RUNS_ROOT"` directory and removed it at the end; every such output ends with `cleanup exists=no`. The largest file written on the host disk was a few bytes. The disk-cap probes wrote only into tmpfs.
- Status: these are probes of mechanisms, not the P0.16 acceptance tests. The findings of probes A to K are [MEASURED] for the compositions shown here; probe L's are exploratory and not [MEASURED]. P0.16 is accepted only when its remote tests pass from a clean commit.

## Question

Which mechanisms available to the unprivileged user joseph_ufl give the P0.16 acceptance properties R1 to R7? The allowed means are user namespaces, systemd-run --user scopes, prlimit, and the delegated memory and pids cgroup files; root and systemctl are excluded. The seven properties are:

- R1: private /dev.
- R2: every mount read-only except the sandbox's own, checked from /proc/self/mountinfo and failing closed.
- R3: $HOME and the scratch root hidden except the workdir, the harness, and the toolchains.
- R4: stdout and stderr capped, with a truncation flag.
- R5: a workdir disk cap.
- R6: no core with the host handler.
- R7: no survivor after the runner's timeout kill.

The P0.16 implementation (lassi/executors/sandbox.py, lassi/toolchains/_base.py) depends on the answer. So does every later task that runs generated code natively.

Classification: factual (answered by running probes on the host).

## Environment facts (probes A and A2)

rx 20260923-132632-exec-f112 (tail only; the head was cut by rx's 120-line tail) and rx 20260923-132643-exec-2be1:

```
$ uv run tools/rx.py exec --timeout 120 -- 'date -Is; uname -r; unshare --version; timeout --version | head -1; mount --version; echo HOME=$HOME; ...; wc -l < /proc/self/mountinfo; awk ... | sort | uniq -c; ...; cat /proc/sys/kernel/core_pattern; cat /proc/sys/fs/suid_dumpable; grep -v "^#" /etc/systemd/coredump.conf | grep -v "^$"; ls -ld /var/lib/systemd/coredump; cat /proc/sys/kernel/dmesg_restrict; id'
2026-09-23T13:26:43-07:00
6.6.29+main+3.0.0r1-amd64-gio-epilmore-dev+
unshare from util-linux 2.37.2
timeout (GNU coreutils) 8.32
mount from util-linux 2.37.2 (libmount 2.37.2: selinux, smack, btrfs, verity, namespaces, assert, debug)
HOME=/mnt/nvme10/joseph_ufl
LASSI_SCRATCH=/mnt/nvme10/joseph_ufl
LASSI_TOOLCHAINS=/mnt/nvme10/joseph_ufl/toolchains
LASSI_RUNS_ROOT=/mnt/nvme10/joseph_ufl/lassi-runs
TMPDIR=/mnt/nvme10/joseph_ufl/tmp
== mountinfo lines
286
     89 tmpfs rw
     75 overlay rw
     64 nsfs rw
     18 squashfs ro
     13 ext4 rw
     [... 19 more fstypes, all rw, trimmed ...]
== rw count
267
== optional fields
      6 -
    280 shared
[from f112:]
== escapes
0
== fs
nodev	overlay
|/lib/systemd/systemd-coredump %P %u %g %s %t 9223372036854775808 %h %d
2                                   (fs.suid_dumpable)
[Coredump]                          (coredump.conf: all defaults)
drwxr-xr-x 2 root root 98304 Sep 23 10:46 /var/lib/systemd/coredump
1                                   (kernel.dmesg_restrict)
```

- $HOME equals the scratch root. `getent passwd` also gives /mnt/nvme10/joseph_ufl as the home directory (probe D), so there is no second home to hide.
- The host has 286 mounts, 267 of them rw. They include docker and containerd overlays, kubelet secret tmpfs mounts under root-only directories, nsfs mounts, and a stacked autofs plus binfmt_misc at /proc/sys/fs/binfmt_misc. No mount point needs an escape. 280 mounts are shared, and `unshare -m` makes them private by default (unshare(1)).
- The user cannot read dmesg (dmesg_restrict=1).

## R2: making every mount read-only (probe B)

rx 20260923-132823-exec-873e. A per-mount remount loop (V1, V1b) was compared with one recursive `mount_setattr(2)` call (V2), each inside `systemd-run --user --scope --quiet unshare -rinmpfu [--mount-proc]`. Script bodies:

```
# v1.sh: remount every rw mount read-only, keeping its other per-mount flags
printf '%s\n' "$mi" | awk 'substr($6,1,2)=="rw" {o=$6; sub(/^rw/,"ro",o); print o, $5}' | {
  while read -r opts mp; do
    if err=$(mount -o "remount,bind,$opts" "$mp" 2>&1); then ok=$((ok+1)); else fail=$((fail+1)); echo "FAIL $mp ($opts): $err"; fi
  done; echo "v1 remounts ok=$ok fail=$fail"; }
# v2.py: one call, AT_FDCWD, "/", AT_RECURSIVE, attr_set = MOUNT_ATTR_RDONLY
rc = libc.syscall(442, -100, b"/", 0x8000, ctypes.byref(attr), ctypes.sizeof(attr))
```

Output (rx showed the last 120 of 330 lines, so V1 with --mount-proc and the first lines of V1b were lost; the FAIL lines shown were 93 kubelet and containerd paths, trimmed):

```
FAIL /var/snap/microk8s/common/var/lib/kubelet/pods/.../kube-api-access-8zj42 (ro,relatime): mount: ...: cannot mount tmpfs read-only.
FAIL /var/snap/microk8s/common/run/containerd/io.containerd.runtime.v2.task/k8s.io/.../rootfs (ro,relatime): mount: ...: cannot mount overlay read-only.
[... trimmed ...]
v1 remounts ok=135 fail=128
v1 loop ms=1849
v1 after rw=129
  still rw: 17080 17074 /run/docker/netns/e9fc26eaea9f nsfs
  still rw: 17157 17156 /proc/sys/fs/binfmt_misc autofs
  still rw: 17265 17264 /mnt/nvme3/docker-data/overlay2/.../merged overlay
  [... trimmed ...]
v1 stacked mountpoints (same path twice):
/proc/sys/fs/binfmt_misc
rc=0
=== V2 recursive mount_setattr (--mount-proc)
v2 before rw=264
v2 mount_setattr(/, AT_RECURSIVE, RDONLY) rc=0 errno=0  ms=0.8
v2 after rw=0
touch: cannot touch '/mnt/nvme10/joseph_ufl/lassi-runs/probe.SzvDm2/x': Read-only file system
new tmpfs after: writable, 1 /tmp lines, top opts rw,relatime
rc=0
host view after: rw=267; ...
cleanup exists=no
```

Findings [MEASURED]:
- Remounting mount by mount does not work here. 128 of 263 rw mounts refused, mostly paths under root-only directories (the cause is an inference). A stacked mount such as the autofs under binfmt_misc cannot be reached by path at all, and the loop took 1.8 s.
- One `mount_setattr(AT_FDCWD, "/", AT_RECURSIVE, {attr_set=MOUNT_ATTR_RDONLY})` made all 264 rw mounts read-only in 0.8 ms. That includes stacked and covered mounts, locked mounts, /proc, and /sys/fs/cgroup. Mounts made afterwards are writable, and the host's own view stays at 267 rw.
- util-linux 2.37.2 has no recursive read-only option, so the call needs a helper. `python3 -I -S -c <constant>` with ctypes does it (syscall 442; mount_setattr(2): Linux 5.12+).

The fail-closed check (probes I and J): after setup, awk reads /proc/self/mountinfo and fails if any mount with rw per-mount options has a source other than `lassi-*` or an fstype other than tmpfs, devpts, or overlay. The negative control in J2 bound the host harness read-write at /var/tmp before the check. The check printed `writable host mount: /var/tmp ext4 /dev/nvme23n1p1` and exited 1 with no ready marker, and the program did not run. With the same arguments and no break (J2b), the program ran.

## R1: private /dev (probe C)

rx 20260923-133007-exec-e0a0. The setup runs inside `systemd-run --user --scope --quiet unshare -rinmpfu --mount-proc`, after the recursive read-only call:

```
mount -t tmpfs -o size=1m,mode=0700 lassi-stage /tmp
mkdir /tmp/dev
mount -t tmpfs -o size=64k,mode=0755,nosuid,noexec lassi-dev /tmp/dev
for n in null zero full random urandom tty; do touch "/tmp/dev/$n"; mount --bind "/dev/$n" "/tmp/dev/$n"; done
mkdir /tmp/dev/pts /tmp/dev/shm
mount -t devpts -o newinstance,ptmxmode=0666,mode=0620 lassi-devpts /tmp/dev/pts
mount -t tmpfs -o size=64m,mode=1777,nosuid,nodev lassi-shm /tmp/dev/shm
ln -s /proc/self/fd /tmp/dev/fd; [stdin, stdout, stderr likewise]; ln -s pts/ptmx /tmp/dev/ptmx
mount -o remount,bind,ro,nosuid,noexec /tmp/dev
mount --move /tmp/dev /dev
umount /tmp; mount -t tmpfs -o size=64m,mode=1777 lassi-tmp /tmp
exec setpriv --no-new-privs --inh-caps=-all --bounding-set=-all -- sh "$1/check.sh"
```

Output (trimmed):

```
host /dev entries: 365
== /dev inside (find)
/dev/fd /dev/full /dev/null /dev/ptmx /dev/pts /dev/pts/ptmx /dev/random /dev/shm /dev/stderr /dev/stdin /dev/stdout /dev/tty /dev/urandom /dev/zero
write /dev/null ok
zero:  00 00 00 00
urandom bytes: 8
random bytes: 8
write /dev/full failed as expected
cat: /dev/tty: No such device or address
openpty ok: /dev/pts/0
shm write ok
touch: cannot touch '/dev/newfile': Read-only file system
== rw mounts now (source fstype mountpoint)
  rw: lassi-devpts devpts /dev/pts
  rw: lassi-shm tmpfs /dev/shm
  rw: lassi-tmp tmpfs /tmp
rc=0
```

Findings [MEASURED]:
- Inside the sandbox, /dev holds only the allowed nodes, the private devpts instance, shm, and the standard symlinks. The host has 365 entries.
- Host device nodes bound onto files in the tmpfs keep working even though their bind mounts are read-only: writing /dev/null works, and /dev/full gives ENOSPC. `os.openpty()` gets /dev/pts/0 from the new instance.
- /dev/tty has no controlling terminal (ENXIO).
- The /dev tmpfs itself is read-only.

## R3 and R5: hidden view, capped overlay workdir, copy-back (probe D)

rx 20260923-133159-exec-9a60. Layout: $D/run1/trial (the workdir, holding `input`), $D/run2/other/secret (another trial), and $D/harness/h. The setup, after the recursive read-only call:

```
mount -t tmpfs -o size=1m,mode=0700 lassi-stage /tmp
mount --bind "$W" /tmp/s/lo                      # overlay lower (read-only)
mount --bind "$W" /tmp/s/rw; mount -o remount,bind,rw /tmp/s/rw   # copy-back target (probe D only; see I)
mount --bind "$H" /tmp/s/h; mount --rbind "$T" /tmp/s/t
mount -t tmpfs -o size=8m,nr_inodes=1024,mode=0755 lassi-wd /tmp/s/wd; mkdir /tmp/s/wd/upper /tmp/s/wd/work
mount -t overlay -o userxattr,lowerdir=/tmp/s/lo,upperdir=/tmp/s/wd/upper,workdir=/tmp/s/wd/work lassi-workdir /tmp/s/ov
mount -t tmpfs -o size=1m,mode=0755 lassi-hide "$S"         # $HOME equals $S here
mkdir -p "$W" "$H" "$T"
mount --move /tmp/s/rw "$W"; mount --move /tmp/s/ov "$W"; mount --move /tmp/s/h "$H"; mount --move /tmp/s/t "$T"
mount -o remount,bind,ro "$S"
mount -t tmpfs -o size=64m,mode=1777 lassi-tmp /tmp
[program under setpriv; then: cd /; umount /tmp; umount "$W"; python3 copy-back of regular files from /tmp/s/wd/upper]
```

Output (the host file list is trimmed):

```
/mnt/nvme10/joseph_ufl                                (getent passwd home)
overlay with userxattr: ok
scratch root listing: lassi-runs toolchains
runs root listing: probe.zMKyzD
cat: /mnt/nvme10/joseph_ufl/lassi-runs/probe.zMKyzD/run2/other/secret: No such file or directory
[three other users' directories; names omitted]     (ls /mnt/nvme10: outside $HOME and scratch, host permissions apply)
toolchains entries: 2
touch: cannot touch '/mnt/nvme10/joseph_ufl/toolchains/x': Read-only file system
h
touch: cannot touch '/mnt/nvme10/joseph_ufl/lassi-runs/probe.zMKyzD/harness/x': Read-only file system
input
workdir write ok
big write stopped: 8380416 bytes
sh: 15: cannot create .../run1/trial/f1015: No space left on device
program status=2
copyback copied=1017 bytes=11 skipped=1
rc=2
host trial dir after: . ./f0 ... ./f1014 ./input ./out.txt ./sub ./sub/n.txt
36	/mnt/nvme10/joseph_ufl/lassi-runs/probe.zMKyzD/run1/trial
out
nested
```

Findings [MEASURED]:
- Hidden view:
  - The scratch root (which is also $HOME) shows only the skeleton of re-exposed paths.
  - Another trial's file is not found.
  - The toolchains root and the harness are readable and read-only, and the workdir is writable.
- Unprivileged overlayfs with `userxattr` mounts on this kernel, with a tmpfs upper layer over a host lower layer.
- Workdir caps:
  - Writes past the tmpfs `size=8m` stop with ENOSPC (8380416 bytes).
  - File creation past `nr_inodes=1024` stops with ENOSPC at f1015.
  - The host workdir grew only by what copy-back wrote.
- Copy-back of the upper layer after the program brought regular files and directories to the host and skipped the symbolic link.

Probe note: dash exits when a redirection on the special builtin `:` fails, so the program stopped at the inode cap. The rw listing that should have followed did not run. Probe I covers it.

## R5: tmpfs usage and the scope's MemoryMax (probe E)

rx 20260923-133226-exec-764d. Inside the scope and namespaces: `mount -t tmpfs -o size=$size lassi-wd /tmp`, then `head -c $((write_mb*1048576)) /dev/zero > /tmp/big`, and the cgroup's memory counters are read.

```
=== E1 tmpfs 16m under MemoryMax=64M, write 32 MiB
tmpfs size=16m memory.max=67108864 current before=1196032
head: error writing 'standard output': No space left on device
head status=1 file bytes=16777216 memory.current=18522112 shmem=16777216
rc=0
=== E2 tmpfs 128m under MemoryMax=32M, write 64 MiB
tmpfs size=128m memory.max=33554432 current before=1593344
Killed
rc=137
```

Findings [MEASURED]:
- tmpfs pages are charged to the run's scope (shmem=16777216).
- A disk cap under the memory limit gives ENOSPC at the cap.
- A disk cap above the free memory gives a memory kill (137, which classify reads as killed) before ENOSPC.

## R6: core dumps (probes F and G)

rx 20260923-133257-exec-4056 (F):

```
ulimit -c soft=0 hard=unlimited
=== coredumpctl list (own entries), last 5
TIME                            PID  UID  GID SIG     COREFILE EXE                                                                        SIZE
Wed 2026-09-23 10:43:18 PDT 3115867 1025 1028 SIGSEGV present  /mnt/nvme10/joseph_ufl/toolchains/nvhpc@24.11/Linux_x86_64/24.11/compilers/bin/tools/nvcpfe 500.1K
Wed 2026-09-23 10:46:21 PDT 3304327 1025 1028 SIGSEGV present  /mnt/nvme10/joseph_ufl/toolchains/nvhpc@24.11/Linux_x86_64/24.11/compilers/bin/tools/nvcpfe 495.4K
=== F1 crash SIGSEGV under prlimit --core=1 in the sandbox namespaces   [sh -c 'kill -SEGV $$' as pid 1]
rc=0
=== F2 crash SIGABRT (python3 os.abort) under prlimit --core=1           [python3 as pid 1]
timeout: the monitored command dumped core
bash: line 8: 977826 Segmentation fault      timeout 30 systemd-run ... unshare -rinmpfu --mount-proc setpriv ... prlimit --core=1 -- python3 -c 'import os; os.abort()'
rc=139
=== coredumpctl entries since 2026-09-23T13:32:57-07:00
Wed 2026-09-23 13:32:59 PDT 977828 1025 1028 SIGSEGV present  /usr/bin/unshare 20.1K
```

Reading F:
- The user can read its own coredumps. Two compiler crashes from the P0.15 probes are stored ("present").
- F1 did not crash: PID 1 of a pid namespace ignores a signal it sends itself.
- In F2 the program was PID 1 under `--core=1`. It died of a forced SIGSEGV (after SIGABRT was ignored; the SIGSEGV is an inference from the exit status 139). util-linux unshare then re-raised the child's signal on itself. unshare had the caller's soft core limit 0, so systemd-coredump stored a core of /usr/bin/unshare.
- Lesson: the core limit must cover the unshare parent too, and the program must never be PID 1.

rx 20260923-133451-exec-7c3d (G). The real chain shape, with PID 1 = timeout and the program as its child:

```
sbx() { core=$1; shift; timeout 30 prlimit --core="$core" -- systemd-run --user --scope --quiet unshare -rinmpfu --mount-proc setpriv --no-new-privs --inh-caps=-all --bounding-set=-all -- prlimit --core="$core" -- timeout 10 "$@"; }
=== limits seen inside the proposed chain
pid=2
CORE     max core file size    1    1 bytes
=== G2 core=1 outer and inner: sh kill -SEGV self (pid 2)
rc=139
=== G3 core=1: python3 os.abort (SIGABRT)
rc=134
=== G4 core=1: python3 null read (SIGSEGV fault)
rc=139
=== coredumpctl since G2
rc=1                                  (no coredumps found)
=== G1 current mechanism, core=0 inner only: sh kill -SEGV self (pid 2)
timeout: the monitored command dumped core
rc=139
=== coredumpctl since G1
Wed 2026-09-23 13:34:57 PDT 1089128 1025 1028 SIGSEGV present  /usr/bin/dash 21.1K
       Storage: /var/lib/systemd/coredump/core.sh.1025.bbf73a3538a944fea96435db2b0e7517.1089128.1790195697000000.zst (present)
     Disk Size: 21.1K
 Control Group: /user.slice/user-1025.slice/user@1025.service/app.slice/run-r9591738ee874497482b0515387b21ae8.scope
```

Findings [MEASURED]:
- `--core=0` (current mechanism): a crashing sandboxed program reaches systemd-coredump, which stores a core on the root filesystem (/var/lib/systemd/coredump). core_pattern passes a fixed limit, so the process's RLIMIT_CORE of 0 is ignored.
- `prlimit --core=1` outermost and innermost: a self-sent SIGSEGV, SIGABRT from abort(), and a real SIGSEGV fault leave no coredumpctl entry, and the exit statuses 139 and 134 pass through.
- The kernel source (fs/coredump.c) aborts a piped core dump when the soft limit is exactly 1 ("we use cprm.limit of 1 here as a special value"). The same code prints two KERN_WARNING lines per aborted dump, which this user cannot read (dmesg_restrict=1), so that part is not measured.

What was checked: `coredumpctl list` and `coredumpctl info` for the user's own uid, which the journal allows without group membership. The user cannot see other users' entries or the kernel log.

## R7: runner kill (probes H and H2)

rx 20260923-133549-exec-7288 read the wrong cgroup: the loop matched the rx session scope, and its cgroup.kill write to that root-owned session scope was refused. Only its heartbeat lines are usable. rx 20260923-133735-exec-a76a repeats the probe with the run's own `app.slice/run-*.scope`. The program starts a `setsid` child that writes a heartbeat file every 0.2 s, then sleeps. The runner is emulated with `setsid ... &`, and its process group is killed after 2 s, as `os.killpg` does.

```
setsid systemd-run --user --scope --quiet -p RuntimeMaxSec=20 -p TimeoutStopSec=1 unshare "$@" setpriv --no-new-privs --inh-caps=-all --bounding-set=-all -- timeout --kill-after=2 12 sh "$D/prog.sh" "$D" &
[current] ... procs before=6 cgroup.kill perms: --w------- joseph_ufl
[current] 0.5 s after kill: scope procs=5
[current] 3 s after kill: heartbeat RUNNING (...), scope procs=5, scope dir exists=yes
[current] scope dir gone after a further 14 x 0.5 s
[kill-child] 0.5 s after kill: scope procs=0
[kill-child] 3 s after kill: heartbeat stopped (...), scope procs=0, scope dir exists=no
[cgroup.kill] cgroup.kill write rc=0
[cgroup.kill] 0.5 s after kill: scope procs=0
[cgroup.kill] 3 s after kill: heartbeat stopped (...), scope procs=0, scope dir exists=no
```

Findings [MEASURED]:
- With the current flags (`-rinmpfu --mount-proc`), killing the runner's process group leaves 5 processes, including the setsid child, running until the inner timeout.
- `unshare --kill-child` (unshare(1): "Combined with --pid this allows for an easy and reliable killing of the entire process tree below unshare") leaves 0 processes within 0.5 s.
- The run scope's own cgroup.kill is owned by joseph_ufl. Writing 1 to it also leaves 0 processes within 0.5 s.

## R4: output caps (probe K)

rx 20260923-134411-exec-89f3. A Python reader drains both pipes with `selectors` in 64 KiB chunks. It keeps at most 1 MiB per stream and counts the rest. The program runs in the scope and namespaces and writes 200 MiB to stdout and 50 MiB to stderr, then exits 5.

```
Python 3.10.12
rc=5 wall=0.2s
stdout: produced=209715200 kept=1048576 truncated=True
stderr: produced=52428800 kept=1048576 truncated=True
runner ru_maxrss_kib=10240
```

Finding [MEASURED]: a draining, capped reader holds no more than the cap per stream (the runner's peak RSS was 10 MiB for 250 MiB of output). The program is never blocked, and its exit status passes through. No change on the host side is needed for R4.

## Composite (probes I and J)

rx 20260923-134021-exec-ebeb (I) and rx 20260923-134113-exec-6366 (J). The whole proposed setup ran as one constant script with positional arguments:

```
prlimit --core=1 -- systemd-run --user --scope --quiet -p MemoryMax=512M -p MemorySwapMax=0 -p TasksMax=256 \
  -p RuntimeMaxSec=40 -p TimeoutStopSec=1 unshare -rinmpfu --mount-proc --kill-child \
  sh setup.sh <workdir> <harness> <toolchains> <disk_mb> <break> <hide1> <hide2> <cpu> <wall> <kill_after> <argv...>
```

setup.sh (in the product, the `break` argument and the timing line go away, the hidden roots become a counted list, and the file becomes SETUP_SCRIPT):

```
set -eu
W=$1; H=$2; T=$3; DISK=$4; BREAK=$5; HIDE1=$6; HIDE2=$7; CPU=$8; WALL=$9; shift 9; KA=$1; shift
for tool in env nice setpriv prlimit timeout python3 awk unshare; do command -v "$tool" > /dev/null; done
RO='import ctypes, sys
libc = ctypes.CDLL(None, use_errno=True)
attr = (ctypes.c_uint64 * 4)(1, 0, 0, 0)
if libc.syscall(442, -100, b"/", 0x8000, attr, 32) != 0:
    sys.exit("mount_setattr: errno %d" % ctypes.get_errno())'
CB='[python: os.walk the upper layer; make directories, skip symbolic links, copy regular files with
shutil.copyfile(follow_symlinks=False); print "copy-back: N files" on stderr]'
echo 0 > /proc/sys/user/max_user_namespaces
python3 -I -S -c "$RO"
mount -t tmpfs -o size=1m,nr_inodes=64,mode=0700 lassi-stage /tmp
mkdir /tmp/s /tmp/s/lo /tmp/s/rw /tmp/s/h /tmp/s/t /tmp/s/wd /tmp/s/ov /tmp/s/dev
mount --bind "$W" /tmp/s/lo
if [ -n "$H" ]; then mount --bind "$H" /tmp/s/h; fi
if [ -n "$T" ]; then mount --rbind "$T" /tmp/s/t; fi
mount -t tmpfs -o "size=${DISK}m,nr_inodes=4096,mode=0755" lassi-wd /tmp/s/wd
mkdir /tmp/s/wd/upper /tmp/s/wd/work
mount -t overlay -o userxattr,lowerdir=/tmp/s/lo,upperdir=/tmp/s/wd/upper,workdir=/tmp/s/wd/work lassi-workdir /tmp/s/ov
mount -t tmpfs -o size=64k,nr_inodes=64,mode=0755,nosuid,noexec lassi-dev /tmp/s/dev
for n in null zero full random urandom tty; do touch "/tmp/s/dev/$n"; mount --bind "/dev/$n" "/tmp/s/dev/$n"; done
mkdir /tmp/s/dev/pts /tmp/s/dev/shm
mount -t devpts -o newinstance,ptmxmode=0666,mode=0620 lassi-devpts /tmp/s/dev/pts
mount -t tmpfs -o size=64m,mode=1777,nosuid,nodev lassi-shm /tmp/s/dev/shm
ln -s /proc/self/fd /tmp/s/dev/fd; ln -s /proc/self/fd/0 /tmp/s/dev/stdin; ln -s /proc/self/fd/1 /tmp/s/dev/stdout; ln -s /proc/self/fd/2 /tmp/s/dev/stderr; ln -s pts/ptmx /tmp/s/dev/ptmx
mount -o remount,bind,ro,nosuid,noexec /tmp/s/dev
mount --move /tmp/s/dev /dev
mount -t tmpfs -o size=1m,nr_inodes=1024,mode=0755 lassi-hide "$HIDE1"
if [ "$HIDE2" != "$HIDE1" ]; then mount -t tmpfs -o size=1m,nr_inodes=1024,mode=0755 lassi-hide "$HIDE2"; fi
mkdir -p "$W"
mount --move /tmp/s/ov "$W"
if [ -n "$H" ]; then mkdir -p "$H"; mount --move /tmp/s/h "$H"; fi
if [ -n "$T" ]; then mkdir -p "$T"; mount --move /tmp/s/t "$T"; fi
mount -o remount,bind,ro "$HIDE1"
if [ "$HIDE2" != "$HIDE1" ]; then mount -o remount,bind,ro "$HIDE2"; fi
for d in /var/tmp /run; do if [ -d "$d" ]; then mount -t tmpfs -o size=64m,mode=1777 lassi-private "$d"; fi; done
mount -t tmpfs -o size=64m,mode=1777 lassi-tmp /tmp
if [ "$BREAK" = 1 ]; then mount --bind "$H" /var/tmp; mount -o remount,bind,rw /var/tmp; fi   # J2 only
awk '{for (i = 7; i <= NF; i++) if ($i == "-") { fs = $(i + 1); src = $(i + 2); break }
  if ($6 ~ /^rw(,|$)/ && !(src ~ /^lassi-/ && (fs == "tmpfs" || fs == "devpts" || fs == "overlay"))) { print "writable host mount: " $5 " " fs " " src > "/dev/stderr"; bad = 1 } }
  END { exit bad }' /proc/self/mountinfo
cd "$W"
echo lassi-sandbox-ready >&2
status=0
unshare --pid --fork -- env -i PATH="$PATH" HOME="$W" LANG=C.UTF-8 TMPDIR=/tmp nice -n 19 setpriv --no-new-privs --inh-caps=-all --bounding-set=-all -- prlimit --cpu="$CPU" --core=1 -- timeout --kill-after="$KA" "$WALL" "$@" || status=$?
cd /
umount /tmp
umount "$W"
mount --bind /tmp/s/lo /tmp/s/rw
mount -o remount,bind,rw /tmp/s/rw
python3 -I -S -c "$CB" /tmp/s/wd/upper /tmp/s/rw
exit "$status"
```

No writable host mount exists while the program runs. Setup makes the read-write bind of the host workdir only after the program's pid namespace has ended, from the read-only lower bind; clearing a read-only flag the sandbox set itself is allowed. The overlay option string holds only constant staging paths, so no argument ever reaches the comma-separated `-o` parser.

I1 output (the program was a shell script in the workdir; see the rx id for its text):

```
=== I1 functional run
setup ms=105 mounts=302
lassi-sandbox-ready
uid=0 pid=2 /dev: fd full null ptmx pts random shm stderr stdin stdout tty urandom zero
scratch: lassi-runs toolchains | other trial: cat: .../run2/other/secret: No such file or directory
toolchains: 2 entries; touch: cannot touch '/mnt/nvme10/joseph_ufl/toolchains/x': Read-only file system
harness: h; touch: cannot touch '.../harness/x': Read-only file system
workdir writes ok
64 MiB write stopped at 33546240 bytes
coredump_filter write: .../prog.sh: 8: cannot create /proc/self/coredump_filter: Read-only file system
max_user_namespaces=0
OSError: [Errno 101] Network is unreachable
Segmentation fault
crash child status=139
copy-back: 3 files
rc=7
host workdir after: . ./hb ./input ./out.txt ./prog.sh ./sub ./sub/n.txt
host hb unchanged after 1 s: yes
coredumpctl since I1: 0 entries ()
```

J output:

```
=== J2 R2 negative control: the host harness bound read-write at /var/tmp before the check
writable host mount: /var/tmp ext4 /dev/nvme23n1p1
rc=1
=== J2b same arguments without the break
setup ms=88 mounts=302
lassi-sandbox-ready
PROGRAM RAN
copy-back: 0 files
rc=0
=== J3 runner kill mid-run (program with a setsid child; pgid killed after 3 s)
setup ms=107 mounts=302
lassi-sandbox-ready
pgid members: 4; run scope=run-rc1f73c4eda6e4e84b7147fe36bda4755.scope procs=8
scope processes: unshare sh unshare timeout sh sh sleep sleep
1 s after kill: scope procs=0 scope exists=no
runner child status=137
```

(I2 and I3 in rx ebeb were invalid controls. I2 bound the overlay, which is a `lassi-*` mount the check allows, and I3 read the session scope. J2 and J3 replace them.)

Findings [MEASURED] for the composite:
- R1, R2, R3, R5, R6, and R7 held together in the composite. R4 was not part of it: probe K measured a separate, head-only reader built on selectors, not the runner that shipped. Setup took 88 to 107 ms (I1 105, J2b 88, J3 107).
- The exit status passes through (7), and the program's setsid child is gone before copy-back, because the nested pid namespace ends with its init (timeout).
- /proc is read-only for the program, so it cannot rewrite coredump_filter or similar files. max_user_namespaces is 0 and the network is unreachable.
- The check fails closed on a writable host mount.
- A runner kill ends every process in the run scope within 1 s.

## Per-requirement result

| Req | Mechanism | Works | Evidence |
| --- | --- | --- | --- |
| R1 | tmpfs /dev built in a staging tmpfs: bind-mounted host null, zero, full, random, urandom, and tty; `devpts -o newinstance,ptmxmode=0666`; tmpfs shm; standard symlinks; remounted read-only and moved onto /dev with `mount --move` | yes | C (e0a0), I1 (ebeb) |
| R2 | one `mount_setattr("/", AT_RECURSIVE, MOUNT_ATTR_RDONLY)` through `python3 -I -S -c`, then an awk check of /proc/self/mountinfo: every rw mount must be a `lassi-*` tmpfs, devpts, or overlay, else setup exits before the ready marker. A per-mount remount loop does not work (128 of 263 refused). | yes | B (873e), J2 and J2b (6366) |
| R3 | stage binds of the workdir, harness, and toolchains; tmpfs (`lassi-hide`) over $HOME and $LASSI_SCRATCH; `mkdir -p` and `mount --move` the exposures back; hide mounts remounted read-only | yes | D (9a60), I1 (ebeb) |
| R4 | runner-side draining reader, keeping a byte cap per stream and discarding the rest, with truncation flags | yes | K (89f3) |
| R5 | overlay (`userxattr`) whose upper layer is a tmpfs with `size=` and `nr_inodes=`; the program runs in a nested pid namespace; afterwards setup copies regular files and directories from the upper layer into a read-write bind made only then | yes; tmpfs use counts toward MemoryMax | D (9a60), E (764d), I1 (ebeb) |
| R6 | `prlimit --core=1` outermost (so unshare is covered too) and innermost; the program is never PID 1; /proc read-only | yes for crashes; residual below | F (4056), G (7c3d), I1 (ebeb) |
| R7 | `unshare --kill-child` on the outer unshare (the runner's killpg kills unshare, whose death kills PID 1 and the pid namespace); cgroup.kill in the run scope also works | yes | H2 (a76a), J3 (6366) |

## Recommended design for P0.16

1. Command: `prlimit --core=1 -- systemd-run --user --scope --quiet -p MemoryMax -p MemorySwapMax=0 -p TasksMax -p RuntimeMaxSec -p TimeoutStopSec=1 unshare -rinmpfu --mount-proc --kill-child sh -c SETUP_SCRIPT sh <positional layout>`. The layout is the workdir, the harness or "", the toolchains root or "", the disk cap in bytes, N and N hidden roots, CPU, wall, kill-after, and argv.
2. SETUP_SCRIPT runs the steps in the composite's order:
   1. Check the tools, now including python3, awk, and unshare.
   2. Write max_user_namespaces while /proc is still writable.
   3. Make every mount read-only recursively.
   4. Stage the binds.
   5. Mount the capped tmpfs and the overlay.
   6. Build the private /dev.
   7. Hide the roots and move the exposures back.
   8. Mount the private /var/tmp, /run, and then /tmp.
   9. Run the mountinfo check.
   10. Print the ready marker.
   11. Run the program in a nested `unshare --pid --fork --kill-child` under `env -i`, `nice`, `setpriv`, `prlimit --cpu --core=1`, and `timeout`.
   12. Copy back, then exit with the program's status.
   The two Python snippets are constants inside the script. Every path still reaches it only as a positional argument.
3. SandboxSpec gains the toolchains root, the hidden roots, and the disk cap. It should normalize the hidden roots, dropping a root that lies under another, because mounting a tmpfs on a path that the outer tmpfs already hid would fail under `set -e`. It should also refuse a toolchains root that is or contains the workdir, for the same reason it refuses such a harness.
4. The runner (lassi/toolchains/_base.py) gets a capped, draining reader with truncation flags on CommandResult. SandboxResult carries them. Whether RunResult and the Trial record carry them needs a check against the bible's Result Record; stop if it needs a record change.
5. Copy-back status: have the setup print a final done line on stderr after copy-back and treat its absence as a copy-back failure. The program's processes are all dead by then, so nothing can print after it. The runner should keep a small tail buffer as well as the capped head, so that line survives truncation.
6. Optional second layer for R7: on its own timeout, the sandbox's runner writes 1 to the run scope's cgroup.kill, found from /proc/<pid>/cgroup before the kill (measured working in H2). It covers the small window before unshare's child sets its parent-death signal. (Not implemented; the window is listed under Known limits in lassi/executors/sandbox.py.)

## Known limits to document (inferences unless marked)

- A program that lowers its own soft RLIMIT_CORE from 1 to 0 before it crashes reaches systemd-coredump. The kernel special-cases only a limit of exactly 1, and G1 measured that a limit of 0 is stored. Not measured for the lowering itself, to avoid storing another core. Setting /proc/self/coredump_filter to 0 before /proc turns read-only would shrink such a core to headers and notes; that is also not measured. (Closed after review: a seccomp filter refuses the lowering, and the setup writes 0 to coredump_filter; see the fixer addendum, probe L, exploratory, and the P0.16 remote tests.)
- Each crash that `--core=1` stops prints two kernel warning lines, "has RLIMIT_CORE set to 1" and "Aborting core" (fs/coredump.c). A crash loop therefore adds kernel log lines. Not observable by this user.
- tmpfs workdir writes count toward MemoryMax [MEASURED, E]. The disk cap should stay well under memory_mb, and a write that exhausts memory first is a memory kill (137, killed), not ENOSPC.
- Copy-back returns regular files and directories only. Symbolic links, devices, FIFOs, and deletions of lower files (whiteouts) are not applied to the host workdir. A rewritten lower file is copied back. Native executor output_files already lists only new regular files.
- Host paths outside $HOME and the scratch root keep host read permissions (for example /mnt/nvme10 lists other users' directories [MEASURED, D]), and /sys stays visible read-only. R3 covers $HOME and the scratch root only.
- The program's /proc belongs to the outer pid namespace, so it can see PID 1 (the setup shell, which holds capabilities in the user namespace). Kernel ptrace rules (a caller without the target's capabilities) should deny it /proc/1/fd, environ, and ptrace, and PID 1 ignores signals from inside its namespace. Not measured.
- Setup now depends on /usr/bin/python3 (run with -I -S, so no PYTHON* variable or user site can inject code). The syscall number 442 is the same on x86_64 and the generic table.

## Side effects of the probes on the host

- F2 made systemd-coredump store one core of /usr/bin/unshare (20.1K) under /var/lib/systemd/coredump. This was unintended: the program ran as PID 1 and unshare re-raised the signal.
- G1 stored one core of /usr/bin/dash (21.1K): the requested check of the current `--core=0` mechanism.
- Both files are root-owned on the root filesystem, and this user cannot delete them. systemd's tmpfiles rule normally ages coredumps out after 3 days; that rule was not verified on alpha01.
- The two nvcpfe cores (500.1K, 495.4K) predate this spike (P0.15 compile probes).
- All four files are queued for the owner as OQ-014 (plans/OWNER-QUEUE.md). No later probe or test crashes a program without prlimit --core=1 around the whole command: the remote R6 test refuses to crash anything otherwise, and it first checks that the program cannot lower its core limit.

## Consequences for the plan

- P0.16 can proceed without root or systemctl, using the design above. Its remote tests should prove each requirement against this spike's findings:
  - R1 by an allow-list of /dev entries. Keep accelerator node names out of rx command lines, because the gate refuses them; the test file may hold them.
  - R2 with a negative control that a test can inject. For example, a setup test that makes /var/tmp a read-write host bind and asserts SandboxUnavailableError.
  - R5 inside tmpfs only, with a cap of a few MiB, never filling the host disk.
  - R6 with SIGSEGV and SIGABRT crashes and `coredumpctl -q list --since` for the user's uid.
  - R7 with a setsid child and the runner's own timeout.
- Handoff, out of P0.16 scope (plans/PHASE-NOTES.md, "P0.20 scope"; task P0.20): compiles run with the host soft core limit 0, which systemd-coredump ignores, so compiler crashes store cores on the root filesystem (the two nvcpfe cores above). Running compiles under `prlimit --core=1` would stop that. This belongs to P0.20, the task that hardens compiles.

## Sources

- mount_setattr(2), https://man7.org/linux/man-pages/man2/mount_setattr.2.html (read 2026-09-23): AT_RECURSIVE "Change the mount properties of the entire mount tree"; Linux 5.12.
- unshare(1), https://man7.org/linux/man-pages/man1/unshare.1.html (read 2026-09-23): --kill-child, and --propagation defaulting to private.
- Linux v6.6 fs/coredump.c, https://raw.githubusercontent.com/torvalds/linux/v6.6/fs/coredump.c (read 2026-09-23): `.limit = rlimit(RLIMIT_CORE)`, the `if (cprm.limit == 1)` abort for piped dumps, and the `__get_dumpable` check.
- Linux overlayfs documentation, https://docs.kernel.org/filesystems/overlayfs.html: the `userxattr` option for unprivileged mounts (not re-read for this spike; the mount was measured in D and I).

## Proposed bible edit (factual; apply once the P0.16 remote tests pass from a clean commit)

Applied on 2026-09-23 in revised form (master revision 81; docs/BIBLE.md, Sandbox bullet and Decision Log). The revision adds the round-3 changes (keyring refusal, hang judged by the program's own run time), the chosen constants as [DESIGN] values, the acceptance run (46 remote tests from commit 59b5799, rx 20260923-173420-desktop-8r113ei-p0-core-192f, results/p0-sandbox-hardening/), and marks the limits that are inferred rather than measured. The text below is the proposal as written before round 3.

In Sandbox, second bullet, replace the final sentence, from "Open gaps: host /dev is visible" to "required before native runs of generated code).", with:

"P0.16 hardening, all unprivileged:
- The whole command runs under prlimit --core=1, because the kernel aborts a piped core dump at that limit and systemd-coredump ignores a limit of 0, and under env -i with a constant PATH, so no process inside holds the caller's environment. unshare gets --kill-child.
- Setup makes every mount read-only with one recursive mount_setattr call, builds a private /dev (tmpfs with bind-mounted null, zero, full, random, urandom, and tty; a new devpts instance; a private shm), hides $HOME, $LASSI_SCRATCH, and the runs root under tmpfs, re-exposing the workdir, the harness, and $LASSI_TOOLCHAINS, and hides /sys device attributes and every /var entry but tmp.
- The workdir is an overlay whose upper layer is a size-capped tmpfs (at most half the memory limit), and the program's file size limit is the same cap. Setup copies the new regular files back to the host workdir after the program's own pid namespace has ended, and copies none when their sizes total more than the cap.
- The program runs in its own pid and IPC namespaces, in a new session with a new session keyring, under a seccomp filter that keeps it from lowering its core limit or changing another process's limits and refuses AF_VSOCK sockets and io_uring.
- Setup fails closed unless every writable mount in /proc/self/mountinfo is one of its own. A failure of the program's chain before the program starts, or of the copy-back after it, is reported as a sandbox failure, never as the program's result.
- The sandbox's runner caps stdout and stderr with truncation flags; the compilers' runners keep all of their output.
- Limits: workdir writes count toward MemoryMax; paths outside $HOME, the scratch root, the runs root, and /var keep host read permissions, and a pathname socket there stays reachable when its permissions allow.
[MEASURED 2026-09-23: probes A to K in plans/spikes/p0-sandbox-hardening.md; P0.16 remote tests <rx id, commit> (PLACEHOLDER until the clean-commit run)]"

Decision Log entry (top of the table; the count sentence becomes "Forty-four decisions have been made: twenty-six on 2026-09-22 and eighteen on 2026-09-23"):

| 2026-09-23 | Sandbox hardening (P0.16), all unprivileged: prlimit --core=1 and env -i with a constant PATH around the whole command; unshare --kill-child; one recursive read-only mount_setattr over every mount with a fail-closed mountinfo check; a private /dev; tmpfs hiding $HOME, the scratch root, and the runs root with the workdir, harness, and toolchains re-exposed, and hiding /sys device attributes and /var; a size-capped overlay workdir (at most half the memory limit, also the file size limit) with a byte-budgeted copy-back after the program's pid and IPC namespaces end; a new session, session keyring, and seccomp filter for the program; capped stdout and stderr in the sandbox's runner | Spike plans/spikes/p0-sandbox-hardening.md, probes A to K (rx 20260923-132632-exec-f112 to 20260923-134411-exec-89f3). Per-mount remounts failed on 128 of 263 mounts, while one mount_setattr made all 264 read-only in 0.8 ms. RLIMIT_CORE=1 kept SIGSEGV, SIGABRT, and a fault out of systemd-coredump, which stored a core at --core=0. --kill-child and cgroup.kill each left no process after a process-group kill, where the current command left 5 running. The composite setup took 88 to 107 ms. The P0.16 remote tests from a clean commit (<rx id, commit>, PLACEHOLDER until that run) are the evidence for the seccomp filter's refusal of the program's attempts to lower its core limit, env -i, coredump_filter 0, the /sys and /var hiding, and --fsize |

## Fixer addendum: probe L (after the review of the P0.16 implementation)

The review found holes in the first implementation: a program could lower its own core limit to 0, sparse files and hard links could make the copy-back write far past the disk cap, the setup looked up commands through the caller's PATH after the program ran, the setup shell kept the caller's environment, the program kept the caller's session keyring and possibly its terminal, device attribute files under /sys and daemon sockets under /var stayed in view, and the runs root was hidden only when it lay under a hidden root. Probe L ran the fixed script on alpha01 before the remote tests.

- Date: 2026-09-23, 15:31 and 15:32 (alpha01 clock, UTC-07:00). The probe embedded the exact SETUP_SCRIPT and CONFINE_PROGRAM text of the uncommitted P0.16 working tree (branch p0-core, base 42d6ab3) in one script and ran it through `rx exec`, in the command shape of sandbox_command: `prlimit --core=1 -- env -i PATH=/usr/sbin:/usr/bin:/sbin:/bin XDG_RUNTIME_DIR=... systemd-run --user --scope --quiet -p MemoryMax=256M ... unshare -rinmpfu --mount-proc --kill-child sh -c "$SETUP" sh <workdir> <harness> "$LASSI_TOOLCHAINS" 4194304 1 "$LASSI_SCRATCH" 40 30 2 <program>`. It is exploratory: a dirty tree, not the acceptance run.
- The programs were small Python and shell checks. Nothing crashed, nothing connected to a socket (the host-side socket list comes from `find -type s`), and the probe's mktemp directory under $LASSI_RUNS_ROOT was removed (`cleanup exists=no`).
- rx 20260923-153159-exec-2a9c: the first run. Its check program stopped at the first refused call, because Python raises ValueError, not OSError, when setrlimit fails with EPERM. Its L2 and L3 matched the second run.
- rx 20260923-153236-exec-7db8: the full run. Output, trimmed to the lines that bear on the findings:

```
== id
uid=1025(joseph_ufl) gid=1028(joseph_ufl) groups=1028(joseph_ufl)
== host sockets under /var /snap /opt /srv (listing only; mode owner:group path)
660 root:root /var/snap/microk8s/common/run/containerd.sock.ttrpc
660 root:microk8s /var/snap/microk8s/common/run/containerd.sock
660 root:root /var/snap/lxd/common/lxd-user/unix.socket
666 root:root /var/lib/haproxy/dev/log
777 root:root /var/lib/amd-metrics-exporter/amdgpu_device_metrics_exporter_grpc.socket
755 root:root /var/lib/kubelet/device-plugins/amd.com_gpu
755 root:root /var/lib/kubelet/device-plugins/kubelet.sock
755 root:root /var/lib/kubelet/device-plugins/rngd.sock
666 root:root /var/spool/postfix/dev/log
host session keyring: 629993953
== L1 checks (disk cap 4 MiB)
rc=0 ms=8346                    (most of it the program's socket walk, cut at 8 s)
lassi-sandbox-ready
lassi-sandbox-done
setrlimit_core_0 = "ValueError not allowed to raise maximum limit"
setrlimit_nofile_same = "ok"
prlimit_self_core_0 = "error EPERM"
prlimit_pid1_nofile = "error EPERM"
prlimit_read_core = "ok"
socket_vsock = "error EPERM"
socket_inet = "ok"
io_uring_setup = [-1, "EPERM"]
sid_pid = [1, 2]
open_dev_tty = "error ENXIO"
coredump_filter = "00000000"
session_keyring = 495102098
rlimit_core = [1, 1]
rlimit_fsize = [4194304, 4194304]
sys_class = []
sys_bus = []
sys_devices_nonempty = []
sys_devices_count = 32
cpu_online = "0-255"
var = {"backups": [], "cache": [], "crash": [], "lib": [], "local": [], "lock": "not-dir", "log": [], "mail": [], "opt": [], "run": ["mount"], "snap": [], "spool": [], "tmp": [], "www": []}
env = ["HOME", "LANG", "PATH", "TMPDIR"]
path = "/usr/sbin:/usr/bin:/sbin:/bin"
proc_environ = [["1", "EACCES", null], ["90", "EACCES", null], ["91", "read", false], ["92", "read", false]]
sockets_walk_cut = "/boot"
sockets_inside_count = 0
== L2 sparse files over the cap
rc=0
PAST_CAP_WRITE error 27
SPARSE_DONE
lassi-sandbox-ready
lassi-sandbox copy-back: the new files total 33554430 bytes, over the 4194304-byte workdir cap; none was copied back
lassi-sandbox-done incomplete
host work bytes: 7964            (the two probe scripts and L1's out.txt)
== L3 shell tries ulimit -c 0 (no crash)
rc=0
REFUSED
0                                (dash prints the core limit in 512-byte blocks; the limit is 1 byte)
/bin/sh: 1: ulimit: error setting limit (Operation not permitted)
lassi-sandbox-done
== coredumpctl since the probe began
No coredumps found.
cleanup exists=no
```

Findings (exploratory, dirty tree; not [MEASURED]) for the composition probe L ran; the remote tests from a clean commit are the acceptance evidence:
- env -i with only PATH and XDG_RUNTIME_DIR is enough for `systemd-run --user --scope`. The program's environment holds HOME, LANG, PATH (the constant), and TMPDIR. It cannot read the environment of the setup shell (pid 1) or of the nested unshare (EACCES), and the probe's secret, set in the caller's environment, appeared in no environment it could read.
- The setup's write of 0 to /proc/self/coredump_filter succeeds inside the user namespace, and the program inherits the value.
- CONFINE_PROGRAM runs. The program's session is led by timeout (sid 1; the program is pid 2), /dev/tty gives ENXIO, and its session keyring differs from the caller's. The seccomp filter refuses, with EPERM, setrlimit on RLIMIT_CORE (Python reports it as ValueError), prlimit64 on RLIMIT_CORE, a limit change of pid 1, an AF_VSOCK socket, and io_uring_setup, and allows the rest. A shell's `ulimit -c 0` is refused, and the core limit stays 1.
- The read-only tmpfs mounts hide /sys/class, /sys/bus, and the contents of every /sys/devices entry but system; the cpu topology still reads 0-255. They hide every /var entry but tmp, and they leave the symbolic links /var/run and /var/lock alone. The program's walk found no socket under /var, /snap, /opt, /srv, /etc, /home, /mnt, /root, or /media before its 8 s budget ran out at /boot.
- On the host, joseph_ufl belongs to no supplementary group. Under /var, the sockets a non-root user can connect to are /var/lib/haproxy/dev/log (666), /var/lib/amd-metrics-exporter/amdgpu_device_metrics_exporter_grpc.socket (777, a GPU metrics exporter), and /var/spool/postfix/dev/log (666). All are hidden now. That generated code could reach all three before this change is an inference from their modes and the earlier view of /var, not probed.
- --fsize at the cap made a write 64 MiB into a file fail with EFBIG (27) while SIGXFSZ was ignored. Eight sparse files of 4 MiB less one byte each fit in the tmpfs, and the copy-back refused all of them. Only the files already on the host remained, and the setup's done line carried the incomplete mark.
- After the private /tmp, /var/tmp, /run, and /dev/shm are unmounted, `mount --bind` and `mount -o remount,bind,rw` still work for the copy-back.
- No core was stored (coredumpctl for the user's uid).

Not probed here: a crash under the filter (the remote R6 test crashes only after it has checked the refusal), hard links (the remote tests), and what a core with coredump_filter 0 would hold.

## Second review round (design changes, no new probe)

A second review asked whether a failure after the ready marker could be recorded as the program's own result, whether the program and the setup's copy-back were isolated from each other, and whether the seccomp filter covered what R6 needs. No probe ran for these changes; their only host evidence is the P0.16 remote tests (tests/executors/test_sandbox_remote.py), and a run of those tests from a dirty tree is exploratory.

- The ready marker now comes from CONFINE_PROGRAM, after the seccomp filter is in place and right before it execs the program's timeout. A failure of env, nice, setpriv, prlimit, or the confinement therefore comes before the marker and is reported as SandboxUnavailableError, where it used to surface as the program's exit status. Only timeout's own failure (125) and a failed exec of timeout can still do so.
- After the program's namespace ends, the setup sets an EXIT trap that prints a line of its own on any failure, and clears it right before the done line. A done line the program printed itself can no longer end stderr when the copy-back then fails without a word (for example, killed by the memory limit). Sandbox.run counts a done line only after a normal exit, and sets workdir_incomplete whenever no done line counts.
- The nested unshare also gets --ipc, so System V IPC objects the program leaves behind die with its IPC namespace instead of staying charged to the run's memory while the copy-back runs. The kernel frees that namespace from a work item, so the memory may still count for a moment (an inference from the kernel source, not measured).
- The copy-back writes each file under a temporary name in its target directory and renames it over the target, so a host file with a hard link to a file outside the workdir is never written through.
- The filter's coverage for R6 was traced by hand: on x86_64 only setrlimit and prlimit64 change a resource limit, the i386 ABI is killed and x32 refused, the hard core limit is 1 so the soft limit cannot rise above it, and PR_SET_DUMPABLE can only disable a dump. No change was needed; the local tests run the filter through a BPF interpreter.

## Third review round (design changes, no new probe)

Three focused red-team passes (seccomp filter, copy-back and workdir, fail-closed paths and runner) found no major issue. The copy-back pass found nothing. The changes made for the rest:

- Every OSError raised while starting the sandbox command now becomes SandboxUnavailableError, and nothing runs unsandboxed. Before, only FileNotFoundError did; a PermissionError or NotADirectoryError escaped as a traceback.
- A hang is decided by the program's own run time. The setup reads /proc/uptime with the shell's `read` builtin right before the nested unshare and right after the copy-back trap line, and the done line carries both readings. Setup and copy-back time no longer let a program that exits 124, 137, or 143 just before the limit pass for a hang. Residual: the chain's own start-up and teardown time, which is not measured, plus 20 ms.
- The seccomp filter also refuses add_key, request_key, and keyctl (EPERM). A key's serial is global, so a host key would likely stay reachable by its serial through a new session keyring. That is inferred from the kernel source, not measured. On this kernel @u and @us resolve per user namespace inside the sandbox. /proc/keys still lists key metadata; that is a Known limit, also an inference.
- An independent review re-traced every jump of the 26-row filter and ran a throwaway BPF evaluator over syscalls 0 to 599, x32, and three architectures, reporting 0 mismatches. That sweep is a review check, not recorded in the repository; the committed test test_confine_filter_refuses_only_what_it_must covers a table of cases.
- Remote suite on alpha01: rx 20260923-170836-desktop-8r113ei-p0-core-36b0, 46 passed. It ran from a dirty-tree snapshot, so it is exploratory and not [MEASURED]; the acceptance run is the one from the P0.16 commit.
