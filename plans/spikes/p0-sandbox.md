# Spike P0.9: sandbox isolation on alpha01

- Task: P0.9 (plans/p0-core.md). Bible: Sandbox; Agent Rules 6 and 7.
- Host date at first probe: 2026-09-23T05:12:13-07:00 (alpha01).
- Local commit when probed: 7f57ac80d12489c18b13ceae3ce13063d6150f08 (branch p0-core). The working tree was dirty only in plans/STATUS.md; every probe used rx exec, which runs from the scratch root without a checkout, so no repository code took part in any result.
- Device: alpha01 host CPU and kernel only. No device commands, no devcheck, no sudo.
- Access: every probe went through `uv run tools/rx.py exec` (scratch root, no checkout). Temp dirs were created with `mktemp -d -p "$TMPDIR"` (TMPDIR=/mnt/nvme10/joseph_ufl/tmp) and removed at the end of each probe; the removal is shown in the output.
- Network probe: one TCP connect to 1.1.1.1:443 with `timeout 5`, once as baseline and once per mechanism.
- All output below is plain ASCII as received; nothing was escaped.

## Question

Which unprivileged mechanism on alpha01 gives the bible's Sandbox properties: (a) runs without privilege, (b) blocks network, (c) enforces memory, CPU, and wall limits, (d) mounts the harness read-only, (e) separate uid or container? P0.10 (sandbox module and the none and native executors) depends on the answer.

Classification: factual (answered by running probes on the host).

## Probe 0: baseline

Gate refusal, recorded as policy (first attempt, which included `systemctl --user is-system-running`):

```
$ uv run tools/rx.py exec --timeout 90 -- 'date -Is; id; ...; systemctl --user is-system-running 2>&1; ...'
rx: gate refused: command refused by policy pattern /\bsystemctl\b/
(exit code 3, no rx id issued)
```

`systemctl` was dropped; the user manager was checked instead through its bus socket and by running `systemd-run --user` (probe 2, 20260923-051255-exec-dee3).

rx id 20260923-051213-exec-364c, rc=0:

```
$ uv run tools/rx.py exec --timeout 90 -- 'date -Is; id; uname -r; head -3 /etc/os-release; cat /proc/sys/kernel/unprivileged_userns_clone 2>&1; cat /proc/sys/user/max_user_namespaces; cat /proc/sys/kernel/apparmor_restrict_unprivileged_userns 2>&1; for c in bwrap unshare apptainer singularity systemd-run prlimit nsjail firejail; do printf "%s: " $c; command -v $c || echo MISSING; done; echo XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR; ls -d /run/user/$(id -u) 2>&1; cat /sys/fs/cgroup/cgroup.controllers 2>&1; cat /proc/self/cgroup; ls /sys/fs/cgroup/user.slice 2>&1 | head; echo TMPDIR=$TMPDIR; timeout 5 bash -c "exec 3<>/dev/tcp/1.1.1.1/443" && echo CONNECTED || echo BLOCKED'
2026-09-23T05:12:13-07:00
uid=1025(joseph_ufl) gid=1028(joseph_ufl) groups=1028(joseph_ufl)
6.6.29+main+3.0.0r1-amd64-gio-epilmore-dev+
PRETTY_NAME="Ubuntu 22.04.5 LTS"
NAME="Ubuntu"
VERSION_ID="22.04"
cat: /proc/sys/kernel/unprivileged_userns_clone: No such file or directory
12347780
cat: /proc/sys/kernel/apparmor_restrict_unprivileged_userns: No such file or directory
bwrap: MISSING
unshare: /usr/bin/unshare
apptainer: MISSING
singularity: MISSING
systemd-run: /usr/bin/systemd-run
prlimit: /usr/bin/prlimit
nsjail: MISSING
firejail: MISSING
XDG_RUNTIME_DIR=/run/user/1025
/run/user/1025
cpuset cpu io memory hugetlb pids rdma misc
0::/user.slice/user-1025.slice/session-1564.scope
cgroup.controllers
cgroup.events
cgroup.freeze
cgroup.kill
cgroup.max.depth
cgroup.max.descendants
cgroup.pressure
cgroup.procs
cgroup.stat
cgroup.subtree_control
TMPDIR=/mnt/nvme10/joseph_ufl/tmp
CONNECTED
```

Baseline [MEASURED]: the host reaches 1.1.1.1:443 (CONNECTED), so a BLOCKED result inside a mechanism is meaningful. Unprivileged user namespaces are not restricted by a sysctl (no Debian or AppArmor knob present, max_user_namespaces=12347780). cgroup v2 unified hierarchy.

## Probe 1: unshare (util-linux 2.37.2), plus tool inventory

rx id 20260923-051252-exec-b576, rc=0. Command (passed to `rx exec --timeout 120 --` as one string):

```
set -u
D=$(mktemp -d -p "$TMPDIR" sbx.XXXX); echo hi > "$D/f"; echo "D=$D"
unshare --version
echo "== extra tools"; for c in podman docker newuidmap newgidmap python3; do printf "%s: " $c; command -v $c || echo MISSING; done
grep -c "^joseph_ufl:" /etc/subuid /etc/subgid 2>&1
ls /opt 2>&1 | head -20
echo "== (a)+(e) unshare -rn id"; timeout 20 unshare -rn id; echo rc=$?
echo "== (b) unshare -rn connect"; timeout 20 unshare -rn timeout 5 bash -c 'exec 3<>/dev/tcp/1.1.1.1/443' && echo CONNECTED || echo BLOCKED
echo "== (b) unshare -rn ip link"; timeout 20 unshare -rn ip -o link 2>&1 | head -3
echo "== (d) unshare -rm read-only bind"
timeout 20 unshare -rm bash -c "mount --bind $D $D; echo bind_rc=\$?; mount -o remount,bind,ro $D; echo remount_rc=\$?; cat $D/f; touch $D/x; echo touch_rc=\$?"
echo "outside after: $(ls $D | tr '\n' ' ')"
echo "== combined unshare -rnmpf --mount-proc"
timeout 20 unshare -rnmpf --mount-proc bash -c "mount --bind $D $D && mount -o remount,bind,ro $D; id; echo nprocs=\$(ls /proc | grep -c '^[0-9]'); touch $D/y 2>&1; echo touch_rc=\$?; timeout 5 bash -c 'exec 3<>/dev/tcp/1.1.1.1/443' && echo CONNECTED || echo BLOCKED"
echo "rc=$?"
rm -rf "$D"; echo "cleaned: $(ls -d $D 2>&1)"
```

Output:

```
D=/mnt/nvme10/joseph_ufl/tmp/sbx.OTEc
unshare from util-linux 2.37.2
== extra tools
podman: MISSING
docker: /usr/bin/docker
newuidmap: /usr/bin/newuidmap
newgidmap: /usr/bin/newgidmap
python3: /usr/bin/python3
/etc/subuid:1
/etc/subgid:1
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
== (a)+(e) unshare -rn id
uid=0(root) gid=0(root) groups=0(root)
rc=0
== (b) unshare -rn connect
bash: connect: Network is unreachable
bash: line 1: /dev/tcp/1.1.1.1/443: Network is unreachable
BLOCKED
== (b) unshare -rn ip link
1: lo: <LOOPBACK> mtu 65536 qdisc noop state DOWN mode DEFAULT group default qlen 1000\    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
== (d) unshare -rm read-only bind
bind_rc=0
remount_rc=0
hi
touch: cannot touch '/mnt/nvme10/joseph_ufl/tmp/sbx.OTEc/x': Read-only file system
touch_rc=1
outside after: f 
== combined unshare -rnmpf --mount-proc
uid=0(root) gid=0(root) groups=0(root)
nprocs=4
touch: cannot touch '/mnt/nvme10/joseph_ufl/tmp/sbx.OTEc/y': Read-only file system
touch_rc=1
bash: connect: Network is unreachable
bash: line 1: /dev/tcp/1.1.1.1/443: Network is unreachable
BLOCKED
rc=0
cleaned: ls: cannot access '/mnt/nvme10/joseph_ufl/tmp/sbx.OTEc': No such file or directory
```

## Probe 2: systemd-run --user (systemd 249)

rx id 20260923-051255-exec-dee3, rc=0. Command:

```
systemd-run --version | head -1
echo "== delegated controllers"; cat /sys/fs/cgroup/user.slice/user-1025.slice/user@1025.service/cgroup.controllers 2>&1
cat /sys/fs/cgroup/user.slice/user-1025.slice/user@1025.service/cgroup.subtree_control 2>&1
echo "DBUS_SESSION_BUS_ADDRESS=${DBUS_SESSION_BUS_ADDRESS:-unset}"; ls -l /run/user/1025/bus 2>&1
echo "== systemd-run --user --scope true"
timeout 20 systemd-run --user --scope -p MemoryMax=64M -p CPUQuota=50% -p RuntimeMaxSec=5 true 2>&1; echo rc=$?
echo "== systemd-run --user --scope memory hog"
timeout 20 systemd-run --user --scope -p MemoryMax=64M -p MemorySwapMax=0 python3 -c "b=bytearray(256*1024*1024); print('ALLOC OK')" 2>&1; echo rc=$?
echo "== systemd-run --user --scope wall"
timeout 20 systemd-run --user --scope -p RuntimeMaxSec=2 sleep 10 2>&1; echo rc=$?
echo "== systemd-run --user network"
timeout 20 systemd-run --user --scope -p PrivateNetwork=yes timeout 5 bash -c 'exec 3<>/dev/tcp/1.1.1.1/443' 2>&1 && echo CONNECTED || echo BLOCKED
```

Output:

```
systemd 249 (249.11-0ubuntu3.22)
== delegated controllers
memory pids
memory pids
DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1025/bus
srw-rw-rw- 1 joseph_ufl joseph_ufl 0 Sep 23 05:12 /run/user/1025/bus
== systemd-run --user --scope true
Running scope as unit: run-rca31f0a879064ef6a720cbd19647488c.scope
rc=0
== systemd-run --user --scope memory hog
Running scope as unit: run-r9a1d41acd45045d9b60920909c36f5af.scope
bash: line 8: 3800121 Killed                  timeout 20 systemd-run --user --scope -p MemoryMax=64M -p MemorySwapMax=0 python3 -c "b=bytearray(256*1024*1024); print('ALLOC OK')" 2>&1
rc=137
== systemd-run --user --scope wall
Running scope as unit: run-r1820904cf92d40c2981b603910401eb7.scope
Terminated
rc=143
== systemd-run --user network
Unknown assignment: PrivateNetwork=yes
BLOCKED
```

Reading the network line: the BLOCKED printed here is NOT a network block. systemd-run rejected `PrivateNetwork=yes` for a scope ("Unknown assignment") and never ran the connect; the `|| echo BLOCKED` fired on that failure. systemd-run --user gives no network isolation by itself.

Reading the wall line: `sleep 10` would exit 0 after 10 s and the outer `timeout 20` would not fire, so rc=143 (SIGTERM) means RuntimeMaxSec=2 stopped the scope. That is an inference from the rc; elapsed time was not recorded.

## Probe 3: limits, availability checks, combined mechanism, mount layout

rx id 20260923-051328-exec-4cad, rc=0. Command:

```
S=$(mktemp -d -p "$TMPDIR" sbx.XXXX); mkdir -p "$S/harness" "$S/trial"; echo h > "$S/harness/h"
echo "== pkgs"; dpkg -l 2>/dev/null | grep -i -E "apptainer|singularity|bubblewrap" || echo "no apptainer/singularity/bubblewrap package"
ls /opt /usr/local/bin 2>/dev/null | grep -i -E "apptainer|singularity|bwrap" || echo "none under /opt or /usr/local/bin"
echo "== subuid range lines: $(grep -c '^joseph_ufl:' /etc/subuid)"; ls -l /var/run/docker.sock 2>&1
echo "== mem prlimit --as 256M"; timeout 20 unshare -rn prlimit --as=268435456 python3 -c "b=bytearray(1<<30); print('ALLOC OK')" 2>&1 | tail -1; echo rc=${PIPESTATUS[0]}
echo "== cpu prlimit --cpu=1"; timeout 20 unshare -rn prlimit --cpu=1 bash -c 'while :; do :; done'; echo rc=$?
echo "== wall timeout 2"; timeout 2 unshare -rn sleep 10; echo rc=$?
echo "== scope cgroup files"; timeout 20 systemd-run --user --scope -p MemoryMax=64M -p CPUQuota=50% bash -c 'c=$(cut -d: -f3 /proc/self/cgroup); echo cg=$c; echo memory.max=$(cat /sys/fs/cgroup$c/memory.max); ls /sys/fs/cgroup$c/cpu.max 2>&1' 2>&1
echo "== scope+unshare memory hog"; timeout 20 systemd-run --user --scope -p MemoryMax=64M -p MemorySwapMax=0 unshare -rnmpf --mount-proc python3 -c "b=bytearray(256*1024*1024); print('ALLOC OK')" 2>&1; echo rc=$?
echo "== scope+unshare net"; timeout 20 systemd-run --user --scope -p MemoryMax=64M unshare -rn timeout 5 bash -c 'exec 3<>/dev/tcp/1.1.1.1/443' 2>&1 && echo CONNECTED || echo BLOCKED
echo "== layout: scratch ro, trial rw, harness ro"
timeout 20 unshare -rmn bash -c "mount --bind $S/trial $S/trial; mount --rbind $S $S; mount -o remount,bind,ro $S; mount --bind $S/harness $S/harness; mount -o remount,bind,ro $S/harness; touch $S/top 2>&1; echo top_rc=\$?; touch $S/harness/w 2>&1; echo harness_rc=\$?; touch $S/trial/ok; echo trial_rc=\$?"
echo "outside: $(cd $S && find . -type f | sort | tr '\n' ' ')"
rm -rf "$S"; echo "cleaned: $(ls -d $S 2>&1)"
```

Output:

```
== pkgs
no apptainer/singularity/bubblewrap package
none under /opt or /usr/local/bin
== subuid range lines: 1
srw-rw---- 1 root docker 0 Sep 22 09:34 /var/run/docker.sock
== mem prlimit --as 256M
MemoryError
rc=1
== cpu prlimit --cpu=1
bash: line 6: 3830278 Killed                  timeout 20 unshare -rn prlimit --cpu=1 bash -c 'while :; do :; done'
rc=137
== wall timeout 2
rc=124
== scope cgroup files
Running scope as unit: run-r7bca05f14fcd4ca4b939ed5306c2caeb.scope
cg=/user.slice/user-1025.slice/user@1025.service/app.slice/run-r7bca05f14fcd4ca4b939ed5306c2caeb.scope
memory.max=67108864
ls: cannot access '/sys/fs/cgroup/user.slice/user-1025.slice/user@1025.service/app.slice/run-r7bca05f14fcd4ca4b939ed5306c2caeb.scope/cpu.max': No such file or directory
== scope+unshare memory hog
Running scope as unit: run-r7b83188178b0484a921d08b23c249bde.scope
bash: line 9: 3833140 Killed                  timeout 20 systemd-run --user --scope -p MemoryMax=64M -p MemorySwapMax=0 unshare -rnmpf --mount-proc python3 -c "b=bytearray(256*1024*1024); print('ALLOC OK')" 2>&1
rc=137
== scope+unshare net
Running scope as unit: run-r31affa67f3184388a7c8d92efe0b26d7.scope
bash: connect: Network is unreachable
bash: line 1: /dev/tcp/1.1.1.1/443: Network is unreachable
BLOCKED
== layout: scratch ro, trial rw, harness ro
touch: cannot touch '/mnt/nvme10/joseph_ufl/tmp/sbx.1fW8/top': Read-only file system
top_rc=1
touch: cannot touch '/mnt/nvme10/joseph_ufl/tmp/sbx.1fW8/harness/w': Read-only file system
harness_rc=1
trial_rc=0
outside: ./harness/h ./trial/ok 
cleaned: ls: cannot access '/mnt/nvme10/joseph_ufl/tmp/sbx.1fW8': No such file or directory
```

## Per-mechanism results

yes, no, partial, or n/a. Evidence names the probe and rx id.

| Mechanism | (a) unprivileged | (b) blocks network | (c) memory, CPU, wall limits | (d) harness read-only | (e) separate uid or container |
| --- | --- | --- | --- | --- | --- |
| bubblewrap (`bwrap`) | n/a: not installed | n/a | n/a | n/a | n/a |
| `unshare -rn` (util-linux 2.37.2) | yes: runs as uid 1025 with no sudo; `unshare -rn id` rc=0 (probe 1, 051252-b576) | yes: "connect: Network is unreachable", BLOCKED; only `lo`, state DOWN (probe 1) | no by itself (namespaces carry no limits); yes when combined with prlimit and timeout: `prlimit --as=256M` gave MemoryError rc=1, `prlimit --cpu=1` busy loop Killed rc=137, `timeout 2 ... sleep 10` rc=124 (probe 3, 051328-4cad) | yes with `-m`: bind plus `remount,bind,ro` gives "Read-only file system", touch_rc=1; host dir unchanged (probe 1) | partial: user, net, mount, and pid namespaces (inside uid 0, nprocs=4 with `-p --mount-proc`), but the host uid stays 1025, not a separate host uid (probe 1) |
| Apptainer or Singularity | n/a: not installed (no binary, no dpkg package, nothing under /opt or /usr/local/bin) | n/a | n/a | n/a | n/a |
| `systemd-run --user --scope` (systemd 249) | yes: user manager reachable at /run/user/1025/bus; scope created, rc=0 (probe 2, 051255-dee3) | no: "Unknown assignment: PrivateNetwork=yes"; the connect never ran (probe 2) | partial: memory yes (MemoryMax=64M, MemorySwapMax=0 hog Killed rc=137; memory.max=67108864 in the scope); wall yes (RuntimeMaxSec=2 on sleep 10 gave rc=143); CPU quota no: only `memory pids` are delegated and the scope has no cpu.max, so CPUQuota=50% is accepted but not enforced (probes 2 and 3) | no: no mount namespace | no: same uid, not a container |

Other options seen, not usable unprivileged [MEASURED]: docker is installed, but /var/run/docker.sock is `root docker 0660` and uid 1025 is not in group docker. podman, nsjail, and firejail are missing.

## Chosen Mechanism

The chosen mechanism is `systemd-run --user --scope` for cgroup memory and wall limits, wrapping `unshare -rnmpf --mount-proc` for isolation, with `prlimit --cpu` for CPU time and an outer `timeout` as a second wall stop. Per trial:

```
timeout <wall_s + grace> systemd-run --user --scope --quiet \
    -p MemoryMax=<mem> -p MemorySwapMax=0 -p RuntimeMaxSec=<wall_s> -p TasksMax=<n> \
  unshare -rnmpf --mount-proc sh -c '<bind trial dir rw; rbind scratch root, remount ro; bind harness, remount ro; exec prlimit --cpu=<cpu_s> -- <cmd>>'
```

Evidence [MEASURED]:
- Network: unshare net namespace BLOCKED inside and outside the scope (probes 1 and 3), against a CONNECTED baseline (probe 0).
- Memory: a 256 MiB allocation under the 64M scope plus the namespaces was Killed, rc=137 (probe 3).
- Wall: RuntimeMaxSec gave rc=143 (probe 2); `timeout` gave rc=124 (probe 3).
- CPU: `prlimit --cpu=1` killed a busy loop, rc=137 (probe 3). This caps CPU time, not CPU share.
- Read-only harness: the mount layout (scratch root read-only, harness read-only, trial dir writable) gave top_rc=1, harness_rc=1, trial_rc=0, and the host saw only the trial write (probe 3).
- Unprivileged: every probe ran as uid 1025 with no sudo.

Inferences, not measured:
- TasksMax should work because the pids controller is delegated. It was not exercised.
- RLIMIT_AS (`prlimit --as`) works, but it counts virtual address space. CUDA and JIT runtimes reserve large virtual ranges, so cgroup MemoryMax is the memory limit and RLIMIT_AS is only a fallback when the user manager is unavailable.
- The `-p` pid namespace with `--mount-proc` hides host processes (nprocs=4 inside), and `-f` is required with it.
- Under `unshare -r` the sandboxed code keeps uid 1025's write access to every path that is not remounted read-only. The sandbox module must therefore remount read-only everything writable except the trial dir: at least the scratch root (/mnt/nvme10/joseph_ufl) and $HOME. Probe 3 showed the rbind-then-remount pattern works on a temp dir; applying it to the real scratch root and $HOME was not probed.
- A separate host uid would need newuidmap with the subuid range (one /etc/subuid line exists for joseph_ufl, and newuidmap is installed). util-linux 2.37.2 predates the `--map-user` and `--map-auto` options (added in 2.38, from memory, not checked), so this needs a small helper and was not tested.

## Gaps against the bible Sandbox text

1. "cgroup limits on CPU": cgroup CPU quota is not available unprivileged. `memory pids` are the only controllers delegated to user@1025.service [MEASURED]. Delegating cpu needs root: a systemd drop-in `Delegate=cpu cpuset io memory pids` for user@.service. The measured substitute is RLIMIT_CPU (CPU-time cap) plus wall time. Getting a CPU share quota is an owner access item.
2. "Separate uid or container": the result is a namespace container, meaning user, pid, mount, and net namespaces with root mapped to uid 1025 inside. The host uid is unchanged [MEASURED]. This meets the "or container" branch only if a user-namespace container counts as a container. That is a reading of the rule, so it is flagged for the owner (OQ-011) rather than decided here.
3. The gate refuses `systemctl` (pattern /\bsystemctl\b/), so unit state cannot be inspected through rx. systemd-run itself is allowed.

## Consequences

- Bible Sandbox section: recorded in docs/BIBLE.md (Sandbox) and the 2026-09-23 Decision Log entry.
- P0.10 is unblocked, because network isolation works (unshare net namespace). No owner item is needed for the network. `lassi/executors/sandbox.py` should:
  - build the command above;
  - map exit codes: 124 or 143 means wall time, so set `hang`; 137 under the scope means the memory or CPU limit killed the run; 152 would be SIGXCPU if the soft CPU limit is set below the hard one;
  - fall back to `prlimit --as` for memory, with a logged warning, when /run/user/$UID/bus is missing;
  - fail with a clear message if `unshare -rn` fails, and never run unsandboxed.
- P0.10 remote tests: the four checks in P0.10 (connect fails, memory hog killed, sleep past wall sets hang, write to the harness mount fails) have direct equivalents above, so they are expected to pass on alpha01.
- Apptainer and bubblewrap: neither is installed. Installing either needs root, so there is no reason to request it.

## Addendum (P0.10): end-to-end probe of the composite command

rx id 20260923-052616-exec-d654 (`rx exec`, no checkout; no repository code took part), 2026-09-23T05:26:16-07:00. The script below was sent base64-encoded to `$TMPDIR/sbx-probe.sh`, run with bash, and removed; its work directory under `$TMPDIR` was removed too (`exists=no`).

```
#!/usr/bin/env bash
# P0.10 end-to-end probe of the P0.9 sandbox composite, run through rx exec on alpha01.
# Scratch only: everything lives under $TMPDIR and is removed at the end.
set -u
date -Is
W="$TMPDIR/sbx-probe.$$"
mkdir -p "$W/harness" "$W/trial"
echo harness > "$W/harness/f"

# sandbox <wall_s> <cmd...>: scope limits, namespaces, read-only layout except trial/, CPU-time cap.
sandbox() {
  local wall="$1"; shift
  timeout $((wall + 10)) systemd-run --user --scope --quiet \
    -p MemoryMax=64M -p MemorySwapMax=0 -p RuntimeMaxSec="$wall" \
    unshare -rnmpf --mount-proc sh -c '
      set -e
      mount --bind "$0/trial" "$0/trial"
      mount --rbind "$0" "$0"
      mount -o remount,bind,ro "$0"
      mount --bind "$0/harness" "$0/harness"
      mount -o remount,bind,ro "$0/harness"
      exec prlimit --cpu=2 -- "$@"' "$W" "$@"
}

echo "=== baseline connect (host)"
timeout 5 bash -c 'exec 3<>/dev/tcp/1.1.1.1/443' 2>/dev/null && echo CONNECTED || echo BLOCKED

echo "=== 1 network inside"
sandbox 10 bash -c 'timeout 5 bash -c "exec 3<>/dev/tcp/1.1.1.1/443" 2>&1 && echo CONNECTED || echo BLOCKED'
echo "rc=$?"

echo "=== 2 writes inside"
sandbox 10 sh -c 'touch "$1/top" 2>/dev/null && echo TOP_WROTE || echo TOP_READONLY; touch "$1/harness/h" 2>/dev/null && echo HARNESS_WROTE || echo HARNESS_READONLY; touch "$1/trial/t" && echo TRIAL_WROTE' sh "$W"
echo "rc=$?"
echo "host sees: $(ls "$W" | tr '\n' ' ')| harness: $(ls "$W/harness" | tr '\n' ' ')| trial: $(ls "$W/trial" | tr '\n' ' ')"

echo "=== 3 memory hog (256 MiB under 64M)"
sandbox 10 python3 -c 'b = bytearray(256 * 1024 * 1024); print("ALLOCATED", len(b))'
echo "rc=$?"

echo "=== 4 wall (sleep 30 under RuntimeMaxSec=3)"
start=$(date +%s)
sandbox 3 sleep 30
echo "rc=$? elapsed_s=$(( $(date +%s) - start ))"

echo "=== 5 cpu (busy loop under prlimit --cpu=2)"
sandbox 20 python3 -c 'while True: pass'
echo "rc=$?"

echo "=== 6 normal run"
sandbox 10 sh -c 'echo OK; exit 3'
echo "rc=$?"

rm -rf "$W"
echo "cleanup rc=$? exists=$( [ -e "$W" ] && echo yes || echo no )"
```

Output (the first line, the `date -Is` stamp 2026-09-23T05:26:16-07:00, is omitted, and the shell's multi-line "Killed" job notices for probes 3 and 5 are trimmed to their first line):

```
=== baseline connect (host)
CONNECTED
=== 1 network inside
bash: connect: Network is unreachable
bash: line 1: /dev/tcp/1.1.1.1/443: Network is unreachable
BLOCKED
rc=0
=== 2 writes inside
TOP_READONLY
HARNESS_READONLY
TRIAL_WROTE
rc=0
host sees: harness trial | harness: f | trial: t
=== 3 memory hog (256 MiB under 64M)
[... line 11: 445476 Killed  timeout $((wall + 10)) systemd-run --user --scope ... (trimmed) ...]
rc=137
=== 4 wall (sleep 30 under RuntimeMaxSec=3)
rc=124 elapsed_s=30
=== 5 cpu (busy loop under prlimit --cpu=2)
[... line 11: 475481 Killed  timeout $((wall + 10)) systemd-run --user --scope ... (trimmed) ...]
rc=137
=== 6 normal run
OK
rc=3
cleanup rc=0 exists=no
```

Findings [MEASURED] for the composite as written above:

- Network: blocked inside, against a connecting host baseline.
- Mount layout: the top directory and the harness are read-only, the trial directory is writable, and the host sees only the trial write.
- Memory: a 256 MiB allocation under MemoryMax=64M is killed (rc 137).
- CPU: a busy loop under `prlimit --cpu=2` is killed (rc 137).
- Normal exit: the command's own status passes through (rc 3).
- Wall time: NOT enforced. With RuntimeMaxSec=3 and an outer `timeout 13`, `sleep 30` ran to completion: the probe returned only after 30 s, with rc 124 from `timeout`. Inference, not measured: neither the outer timeout's signal nor RuntimeMaxSec stopped the process inside the namespaces in this composition; one possible cause is that the command runs as PID 1 of the new pid namespace, which ignores SIGTERM without a handler. This differs from probe 2 above, where RuntimeMaxSec alone stopped a scope (rc 143, inferred).

Consequence for P0.10: enforce wall time inside the namespaces, for example `timeout --kill-after=<grace> <wall>` as the command's innermost wrapper, or kill the whole scope on expiry, and prove it with the remote test "a sleep past wall time sets hang" before relying on it. The bible Sandbox bullet, which lists RuntimeMaxSec and an outer timeout as the wall limit, must be corrected when P0.10 settles the mechanism.

### Second addendum probe: wall time by an innermost timeout

rx id 20260923-053016-exec-2413 (`rx exec`, no checkout), 2026-09-23T05:30:16-07:00. Same composite without RuntimeMaxSec, with `timeout --kill-after=2 <wall>` as the innermost wrapper inside the namespaces; sent base64-encoded to `$TMPDIR`, run, and removed.

```
#!/usr/bin/env bash
# P0.10 second addendum probe: wall time enforced by an innermost timeout inside the namespaces. Scratch only.
set -u
date -Is
W="$TMPDIR/sbx-probe2.$$"
mkdir -p "$W/harness" "$W/trial"
sandbox() {
  local wall="$1"; shift
  timeout $((wall + 10)) systemd-run --user --scope --quiet \
    -p MemoryMax=64M -p MemorySwapMax=0 \
    unshare -rnmpf --mount-proc sh -c '
      set -e
      mount --bind "$0/trial" "$0/trial"
      mount --rbind "$0" "$0"
      mount -o remount,bind,ro "$0"
      mount --bind "$0/harness" "$0/harness"
      mount -o remount,bind,ro "$0/harness"
      wall="$1"; shift
      exec prlimit --cpu=20 -- timeout --kill-after=2 "$wall" "$@"' "$W" "$wall" "$@"
}
echo "=== wall (sleep 30 under inner timeout 3)"
start=$(date +%s); sandbox 3 sleep 30; echo "rc=$? elapsed_s=$(( $(date +%s) - start ))"
echo "=== wall, TERM ignored (trap on TERM, sleep 30)"
start=$(date +%s); sandbox 3 sh -c 'trap "" TERM; sleep 30'; echo "rc=$? elapsed_s=$(( $(date +%s) - start ))"
echo "=== normal run"
sandbox 5 sh -c 'echo OK; exit 3'; echo "rc=$?"
rm -rf "$W"; echo "cleanup rc=$? exists=$( [ -e "$W" ] && echo yes || echo no )"
```

Output (the first line, the date stamp above, is omitted):

```
=== wall (sleep 30 under inner timeout 3)
rc=124 elapsed_s=3
=== wall, TERM ignored (trap on TERM, sleep 30)
rc=124 elapsed_s=5
=== normal run
OK
rc=3
cleanup rc=0 exists=no
```

Findings [MEASURED]: with the innermost `timeout --kill-after=2 3`, `sleep 30` stopped after 3 s (rc 124); a command ignoring SIGTERM stopped after 5 s (rc 124; consistent with the wall time plus the 2 s kill-after grace, an inference, since no signal was captured); a normal exit still passes through (rc 3). Elapsed times come from `date +%s` and are accurate to about 1 s. This is the wall-time form P0.10 should adopt and prove with its remote test.
