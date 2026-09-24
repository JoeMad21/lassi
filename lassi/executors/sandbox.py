"""The sandbox: the only module that runs generated code (Agent Rule 6; bible Sandbox).

Evidence: plans/spikes/p0-sandbox.md (probes 1 to 3 and both addenda)
measured the scope, the namespaces, prlimit --cpu, and the innermost timeout
on the build host (P0.10). plans/spikes/p0-sandbox-hardening.md measured the
P0.16 mechanisms there without root or systemctl, one at a time (probes A to
H and K, inline scripts with no repository code) and as one composite script
(probes I and J). Its fixer addendum ran an uncommitted tree's setup script
on the host; that run is exploratory, not a measurement, so what it saw is
cited as "observed (probe L, exploratory)". Those probes ran earlier forms of
the script below, not this code, so each step says what they covered. The
composition as written here is checked by
tests/executors/test_sandbox_remote.py and counts as proven only once those
tests pass from a clean commit. One command runs each program:

    prlimit --core=1 --
      env -i PATH=<SANDBOX_PATH> [XDG_RUNTIME_DIR=...] [DBUS_SESSION_BUS_ADDRESS=...]
        systemd-run --user --scope --quiet -p MemoryMax=... -p MemorySwapMax=0
            -p TasksMax=... -p RuntimeMaxSec=... -p TimeoutStopSec=1
          unshare -rinmpfu --mount-proc --kill-child
            sh -c SETUP_SCRIPT sh <workdir> <harness> <toolchains> <disk bytes> <N> <hidden roots...>
               <cpu> <wall> <kill-after> [ENVIRONMENT_MARKER <NAME=value...>] <argv...>

- prlimit --core=1 covers every process of the run, unshare included. The
  kernel aborts a core dump piped to the host handler when the soft core
  limit is exactly 1, where alpha01's systemd-coredump stores a core even at
  a limit of 0 (probes F and G).
- env -i starts the sandbox with only PATH=SANDBOX_PATH and the variables
  systemd-run needs to reach the user manager (PASSED_VARIABLES, when set).
  So no process inside, the setup shell included, holds the caller's
  environment (no API key, Agent Rule 12; no LD_PRELOAD or LD_LIBRARY_PATH),
  and every command the setup runs, before and after the program, comes from
  the system directories, never from one the program can write (observed,
  probe L, exploratory).
- systemd-run --user --scope puts the run in its own cgroup, which enforces
  the memory limit (a hog is killed, exit status 137) and sets a task limit
  (TasksMax, not exercised). RuntimeMaxSec is a backstop OUTER_MARGIN_S
  seconds past the wall limit and its kill grace; TimeoutStopSec=1 makes its
  SIGKILL follow one second after its SIGTERM. The backstop is not what
  enforces wall time, and it was not measured in this composition.
- unshare gives the run its own user, IPC, network, mount, pid, and UTS
  namespaces: no network, no view of host processes, and no SysV IPC or
  POSIX message queue shared with the host user's processes (IPC and UTS
  not measured). --kill-child makes the death of unshare kill the setup
  shell, pid 1 of the namespace, and with it every process in it: the
  runner's own timeout kills the process group that holds unshare, so no
  process of the sandbox survives it, not even one in its own session
  (probes H2 and J3).
- SETUP_SCRIPT runs inside the namespaces with `set -eu`, so a failed step
  stops it before the program starts. In order, it:
  1. checks that env, nice, setpriv, prlimit, timeout, python3, awk, and
     unshare are on PATH, and that the disk cap and root count are numbers
     without a leading zero;
  2. writes 0 to /proc/sys/user/max_user_namespaces while it holds its
     capabilities and /proc is writable. That sets the sandbox user
     namespace's own limit (the host value never changes), so the program
     cannot create a nested user namespace and regain capabilities there
     (P0.10). It also writes 0 to its own /proc/self/coredump_filter, which
     every later process inherits, so a core that still reached the host
     handler would hold no memory (the write observed, probe L, exploratory;
     its effect on a core, not measured);
  3. makes every mount read-only with one recursive mount_setattr call
     (READONLY_PROGRAM, run by python3 -I -S), stacked and covered mounts,
     /proc, and the cgroup tree included (probe B). Mounts made afterwards
     are the setup's own and stay writable;
  4. mounts a staging tmpfs on /tmp and binds the workdir (read-only), the
     harness, and the toolchains root (recursively) there;
  5. mounts a tmpfs capped at the disk cap (size=<disk bytes> and 4096
     inodes) and an overlay (userxattr) whose lower layer is the read-only
     workdir bind and whose upper layer is on that tmpfs, so a write past
     the cap fails inside with ENOSPC (probes D and I, with 8 MiB and 1024
     or 4096 inodes). The -o strings hold only constant staging paths and
     the checked number;
  6. builds a private /dev: a tmpfs holding binds of the host's null, zero,
     full, random, urandom, and tty, a new devpts instance, a private shm
     tmpfs, and the links fd, stdin, stdout, stderr, and ptmx; it is
     remounted read-only and moved onto /dev (probe C);
  7. hides each hidden root under a tmpfs, makes the path skeleton to the
     workdir, the harness, and the toolchains root, moves the overlay onto
     the workdir and the harness and toolchains binds back to their paths,
     and remounts each hiding tmpfs read-only (probe D);
  8. mounts an empty read-only tmpfs on /sys/class, /sys/bus, every entry
     of /sys/devices but system, and every entry of /var but tmp (symbolic
     links skipped), so device attribute files and the daemon sockets under
     /var (snap and container runtimes) are out of view (observed,
     probe L, exploratory);
  9. mounts a private tmpfs on /var/tmp and /run when they exist, and on
     /tmp last. The /run mount hides the user and system bus sockets, so
     the program cannot ask a service manager to start anything outside;
  10. fails closed unless every writable mount in /proc/self/mountinfo is
      one of its own (MOUNT_CHECK: a lassi-* tmpfs, devpts, or overlay),
      before the ready marker (probe J2);
  11. reads the clock (/proc/uptime: CLOCK_BOOTTIME cut to 10 ms, read with
      the read builtin, so no tool and no substitution) and runs the
      program in a nested `unshare --pid --ipc --fork --kill-child`, so the
      program is never pid 1, every process it starts dies with that
      namespace, and the System V IPC objects and POSIX message queues it
      leaves behind die with its own IPC namespace instead
      of staying charged to the run's memory while the copy-back runs (the
      composite probes ran it without --ipc and --kill-child; neither was
      measured on its own), under, in order:
      - env -i with only PATH (SANDBOX_PATH), HOME (the workdir), LANG, and
        TMPDIR=/tmp, so no secret in the caller's environment reaches
        generated code (Agent Rule 12);
      - nice -n 19, the lowest CPU priority;
      - setpriv --no-new-privs with empty inheritable and bounding sets: the
        program is uid 0 of its user namespace but holds no capability, so
        it cannot unmount or remount anything, nor raise the limit of step 2;
      - prlimit --cpu (a CPU-time cap; the cgroup cpu controller is not
        delegated, OQ-011), --core=1 again, and --fsize at the disk cap, so
        no file the program writes, sparse or not, grows past the cap
        (EFBIG, or SIGXFSZ when not ignored; observed, probe L, exploratory);
      - CONFINE_PROGRAM (python3 -I -S): a new session, so the program has
        no controlling terminal; a new session keyring, so it holds none of
        the caller's keys; and a seccomp filter that refuses any change of
        RLIMIT_CORE and any limit change of another process, so the
        program cannot lower its core limit to 0, which the host handler
        would honor by storing the core (probe G1); it also refuses AF_VSOCK
        sockets, io_uring, x32 syscalls, and kills any other ABI (observed,
        probe L, exploratory), and refuses the keyring calls add_key,
        request_key, and keyctl: a key's serial is global, so a host key
        (for example in the host user's @u, found through /proc/keys) would
        likely stay reachable by its serial, new session keyring or not (an
        inference from the kernel source, not measured; on this kernel @u
        itself resolves per user namespace). The refusal is what is checked
        (test_the_program_holds_none_of_the_callers_keys, remote). Once the
        filter is in place it prints READY_MARKER on stderr and execs the
        timeout, so the marker follows every step of the chain before the
        program;
      - an innermost `timeout --kill-after`, which is what enforces wall
        time (the spike's second addendum measured exit status 124);
      - only when the spec sets the program's environment
        (SandboxSpec.environment, P0.20): `env -i -- NAME=value...` right
        before the program, so the program gets exactly those variables
        instead of the defaults above, while every tool of the chain, env
        included, still comes from SANDBOX_PATH;
  12. after that namespace has ended, sets an EXIT trap that prints
      "lassi-sandbox: the setup stopped after the program ended" on stderr,
      so any failure from here on ends stderr with a line of its own; reads
      the clock again; unmounts the private /tmp, /var/tmp, /run, and
      /dev/shm (freeing their memory) and the overlay, binds the host
      workdir read-write (the only writable host mount it ever makes, and
      only now), copies the upper layer's new regular files and directories
      into it (COPY_BACK_PROGRAM, under the disk cap and the COPY_DEPTH and
      COPY_PATH limits; not the spike's copy-back), clears the trap, and
      prints DONE_MARKER on stderr with its two clock readings, followed by
      " incomplete" when the copy-back left something out; then it exits
      with the program's status.

The sandbox's default runner, lassi.toolchains capped_runner, caps stdout and
stderr at OUTPUT_CAP_BYTES each (the runner's own timeout line comes on top
after a timeout), keeping the head and the tail, and
SandboxResult carries its truncation flags (probe K measured a head-only
capped reader; this head-and-tail runner is covered by tests). The workdir's
disk cap is SandboxSpec.disk_mb MiB (WORKDIR_DISK_MB unless set), but at most
half of the run's memory limit, since tmpfs pages count toward MemoryMax
(probe E): workdir_cap_bytes.

The script is a constant. Every value reaches it as a positional argument,
so no path or argument is ever parsed by a shell or put in a mount option.
A failure before the ready marker means the program never ran: a missing
tool, a failed mount or write, or a writable host mount stops the script,
and a failure of env, nice, setpriv, prlimit, or CONFINE_PROGRAM (which
prints a "lassi-sandbox confine:" line) ends the chain before the marker.
Sandbox.run reports each as SandboxUnavailableError, so a sandbox that never
started the program is never reported as the program's exit status. So is a
setup that exited normally after the program without its done line (the
copy-back did not finish): the EXIT trap puts a line of the setup's own
after anything the program printed, so a done line the program printed
itself never ends stderr then, and a done line counts only after a normal
exit, since a signal death means the setup died before its own. After a
signal death (the backstop, a memory kill of the setup, or the runner's own
timeout) the result is returned as classified, with workdir_incomplete set.
Only timeout's own failure (status 125), a failed exec of timeout right
after the marker (a "lassi-sandbox confine:" line and status 1), and, when
the spec sets the program's environment, env's own failure (125) or its
failure to run the program (126, or 127 when the program does not exist in
the view, each with an "env:" line) can still surface as the program's exit
status. For compiles the stage runner rules the last one out before the
first build: its --version check runs the pinned compiler through the same
compile runner (lassi.core.runner). Sandbox.run reports a death by signal
N as the shell does, 128 + N. Any OSError the runner raises (the command did
not start) is SandboxUnavailableError too. The hang test compares the
program's own time, the difference of the done line's two clock readings
(SandboxResult.program_s), with the wall limit, so neither setup nor
copy-back time counts; only when no done line counts (the whole sandbox was
killed) does it use the whole command's time (see classify).

Compiles (P0.20). SandboxedCompileRunner is the CommandRunner the stage
runner (lassi.core.runner.build_toolchain) gives every pinned toolchain, so
a compile of model-generated sources runs through this same command, not a
second isolation mechanism: its build dir is the workdir, the pinned
toolchains root is exposed read-only, and $HOME, the scratch root, and the
runs root are hidden roots, so an absolute `#include` under them finds no
file. SandboxSpec.environment carries the compile's environment (PATH,
LANG=C, LC_ALL=C, a private TMPDIR under the build dir, and each variable a
pin names), checked against ENVIRONMENT_NAMES, a fixed allowlist; each
variable reaches the command as one NAME=value element after SETUP_SCRIPT,
so no shell parses it. Program runs keep the defaults and the P0.16
command (environment None). A compile keeps each stream of the compiler's
output whole up to COMPILE_OUTPUT_CAP_BYTES (CappedRunner, far above any
compile output seen, exploratory: 3210 bytes; OUTPUT_CAP_BYTES is for
generated programs), and a line of
the sandbox's own ends its stderr when a stream passed that cap, when a
limit killed the compile, or when its whole sandbox was killed, so nothing
is lost silently. Its wall limit is the toolchain's timeout, and its disk
and memory limits are COMPILE_DISK_MB and COMPILE_MEMORY_MB (see
SandboxedCompileRunner for why).

Setup took 88 to 107 ms in the composite probes (I1, J2b, J3). Not measured:
the backstop, TasksMax, the IPC and UTS namespaces, and the kernel warning
lines that each crash stopped by --core=1 prints (the user cannot read the
kernel log).

Known limits:
- tmpfs pages count toward MemoryMax, so workdir writes (and /tmp, /run,
  /var/tmp, /dev/shm) use the memory limit; workdir_cap_bytes keeps the
  disk cap at half of it, and the copy-back (python3) runs in the other
  half, so a memory limit of a few tens of MiB leaves it little room (an
  inference from the size of a python3 process, not measured).
- The kernel frees the program's IPC namespace, and the System V memory in
  it, from a work item after the namespace's last process has gone, so that
  memory may still count toward MemoryMax for a moment as the copy-back
  starts (an inference from the kernel source, not measured).
- If a core ever reaches the host handler anyway, systemd-coredump stores it
  under /var/lib/systemd/coredump on the alpha01 root filesystem (Agent
  Rule 7), root-owned; the seccomp filter and --core=1 are what keep it out.
- Copy-back returns regular files (with their permission bits, owner
  read-write added, group and other write removed) and directories only;
  symbolic links, devices, FIFOs, and deletions of host files are not
  applied, a host symbolic link is never written through, and an entry
  whose host counterpart has another type is skipped with everything under
  it. Each file is written under a temporary name and renamed over its
  target, so a host file with a hard link to a file outside the workdir (a
  build step could leave one) gets a new inode and the file outside keeps
  its bytes. When the new files total more than the disk cap, nothing is
  copied back, and entries more than COPY_DEPTH levels deep or COPY_PATH
  bytes of path are left out; either way SandboxResult.workdir_incomplete is
  set and the copy-back's "lassi-sandbox copy-back:" line ends stderr.
  Copy-back time counts against the backstop margin; a copy-back cut short
  by it leaves a partial host workdir, perhaps with one .lassi-copy-back.*
  temporary file, reported with workdir_incomplete.
- The disk cap counts the new files' sizes. The host's own blocks for the
  up to 4096 new entries (each directory, and each file's last partial
  block) come on top: at most 16 MiB with 4 KiB blocks (an inference from
  the inode cap, not measured).
- Paths outside $HOME, the scratch root, the runs root, /var, and the
  hidden /sys entries keep the host user's read permission (for example
  other users' directories on /mnt), and a pathname unix socket there stays
  reachable when its permissions allow: connect needs no writable mount,
  and the user's supplementary groups still apply inside. /proc/bus/pci and
  the rest of /sys stay readable.
- The program's /proc is the outer pid namespace's, so it sees the setup
  shell as pid 1. The setup shell's environment holds nothing secret:
  test_the_program_sees_only_the_allowed_environment (remote) checks that no
  environment the program can read holds a secret from the caller's, and
  the program in probe L, exploratory, saw the environ of pid 1 and of the
  nested unshare refused with EACCES. Whether ptrace rules keep the program from pid 1 and
  its capabilities is not measured.
- MOUNT_CHECK trusts a writable mount by its source name (lassi-*) and type,
  not by its mount point; every host mount is read-only before the setup
  mounts anything, so a host mount could pass only by carrying such a name.
- The seccomp filter leaves the rest of an unprivileged user's syscall
  surface open, such as bpf and userfaultfd where the host's sysctls allow
  them. With the keyring calls refused, /proc/keys still lists the host
  user's keys the user may view (type, description, and permissions, not
  their payloads), since procfs is not filtered (an inference from the
  kernel source, not measured).
- program_s starts before the program's chain starts the innermost timeout
  and ends after the chain has ended, and each clock reading is cut to
  10 ms, so a program that stops itself with status 124, 137, or 143 within
  that margin of the wall limit (the chain's start and end, not measured,
  plus up to 20 ms) is reported as a hang.
- The runner kill relies on the parent-death signal that unshare's child
  sets right after it forks; a kill in that short window is not covered (the
  run scope's cgroup.kill, probe H2, is not used).
- The ready, done, and stop lines share stderr with the program. The EXIT
  trap covers every failure the setup shell sees, but a setup shell killed
  by SIGKILL prints nothing; that shows as a signal death, where no done
  line counts.
- Setup needs python3, and the program's chain needs it too, in
  SANDBOX_PATH, and mount_setattr (Linux 5.12 or later); the seccomp filter
  is written for x86_64.
- A compile gets the caller's PATH, but its directories under a hidden root
  show empty inside, so every tool a compiler starts (the host compiler,
  the linker) must come from the toolchains root or from elsewhere on the
  host (on alpha01, /usr/bin). Program runs get SANDBOX_PATH.
"""

from __future__ import annotations

import math
import os
import posixpath
import re
import shutil
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePath, PurePosixPath
from types import MappingProxyType

from lassi.core.interfaces import Limits
from lassi.toolchains import CappedRunner, CommandResult, CommandRunner, capped_runner
from lassi.toolchains._base import OUTPUT_TAIL_BYTES
from lassi.toolchains.pins import PREFIX_VARIABLES

# Grace seconds between the innermost timeout's SIGTERM and its SIGKILL.
KILL_AFTER_S = 2
# Seconds past the wall limit and KILL_AFTER_S before the scope's RuntimeMaxSec backstop fires.
OUTER_MARGIN_S = 5
# Seconds past the backstop before the runner itself gives up on the command (returncode -1).
_RUNNER_MARGIN_S = 10
# The line SETUP_SCRIPT prints on stderr once setup is done, right before it runs the program.
READY_MARKER = "lassi-sandbox-ready"
# The line SETUP_SCRIPT prints on stderr last, once the copy-back of the workdir has finished.
DONE_MARKER = "lassi-sandbox-done"
# What SETUP_SCRIPT adds to the done line when the copy-back left part of the workdir out.
INCOMPLETE_SUFFIX = " incomplete"
# The steps per second of the clock SETUP_SCRIPT reads around the program's namespace: /proc/uptime prints
# CLOCK_BOOTTIME in seconds with two decimals, cut (not rounded) to the step.
UPTIME_STEPS_PER_S = 100
# The documented workdir disk cap in MiB: the size of the tmpfs that holds everything a run writes there.
WORKDIR_DISK_MB = 256
# The copy-back's limits: how many levels deep (path components), and how many bytes of relative path, an entry
# may be.
COPY_DEPTH = 32
COPY_PATH = 1024
# The directories SETUP_SCRIPT covers with a private tmpfs (/dev/shm inside the private /dev); anything under
# them is hidden inside.
PRIVATE_DIRS = ("/tmp", "/var/tmp", "/dev/shm", "/run")
# The directories whose entries SETUP_SCRIPT covers with an empty read-only tmpfs (/var/tmp and
# /sys/devices/system excepted), so no workdir, harness, or toolchains root may lie under them.
SYSTEM_DIRS = ("/var", "/sys")
# The only PATH inside the sandbox, for the setup and the program: the system directories.
SANDBOX_PATH = "/usr/sbin:/usr/bin:/sbin:/bin"
# The caller's environment variables the sandbox command keeps, when set: what systemd-run --user needs.
PASSED_VARIABLES = ("XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS")
# The only names SandboxSpec.environment may hold (P0.20): PATH, the locale (LANG, LC_ALL), TMPDIR, and each
# variable a pin names (lassi.toolchains.pins PREFIX_VARIABLES, NVHPC_CUDA_HOME). HOME, loader variables
# (LD_PRELOAD, LD_LIBRARY_PATH), variables that change a compile silently (NVCC_PREPEND_FLAGS, CPATH), and
# credentials are never on it (Agent Rule 12). It is fixed: nothing adds a name at run time.
ENVIRONMENT_NAMES = frozenset({"PATH", "LANG", "LC_ALL", "TMPDIR", *PREFIX_VARIABLES.values()})
# The positional element that tells SETUP_SCRIPT that the program's environment follows it, one NAME=value
# element per variable, right before the program argv. sandbox_command refuses a program argv that starts with it.
ENVIRONMENT_MARKER = "lassi-sandbox-environment"
# A compile's private temporary directory under its build dir (SandboxedCompileRunner). No model file path may
# start with "@" (lassi.core.files), so no generated file can take its place.
COMPILE_TMPDIR = "@lassi-tmp"
# A compile's limits (SandboxedCompileRunner), chosen with wide margin over what the 14 fixture scenarios and
# one benchmark app (cuda with nvcc, omp with nvc++) needed on alpha01 in an exploratory run
# (plans/spikes/p0-compile-hardening.md; a dirty snapshot, so not a measurement): at most 1.4 s of CPU, 1.7 s of
# wall time, 216 MiB of RSS in one process, 5.5 MiB and 18 entries in TMPDIR, and 1 MiB of outputs.
# The workdir disk cap in MiB: the tmpfs that holds everything the compiler writes in its build dir, its private
# TMPDIR included, and its file size limit.
COMPILE_DISK_MB = 2048
# The memory limit in MiB (MemoryMax, tmpfs pages included): at least twice COMPILE_DISK_MB (here four times), so
# workdir_cap_bytes never halves the disk cap, and a full disk cap still leaves 6 GiB for the compiler itself.
COMPILE_MEMORY_MB = 8192
# The CPU count for the CPU-time cap: each process may use wall_s x COMPILE_CPUS seconds of CPU (1200 s at the
# default 600 s timeout), room for a compiler that runs two threads for the whole wall limit.
COMPILE_CPUS = 2
# A compile's output cap in bytes, for each of stdout and stderr (the CappedRunner SandboxedCompileRunner uses by
# default): 64 MiB, over 20000 times the largest compiler stderr the fixture scenarios and the layout app printed in
# the same exploratory run (3210 bytes), so a compile like those keeps its whole output. It bounds what the host process
# holds, since the runner's buffers lie outside the sandbox's memory limit, and what compile.stderr takes on disk,
# when a generated source makes the compiler print without end (review finding). A stream past it
# keeps its head and its last OUTPUT_TAIL_BYTES bytes, and a "lassi-sandbox:" line at the end of stderr says so.
COMPILE_OUTPUT_CAP_BYTES = 64 << 20
# How much of a failed setup's stderr a SandboxUnavailableError quotes.
_STDERR_TAIL = 2000
# SETUP_SCRIPT's done line: the marker, the two /proc/uptime readings, and the incomplete mark when set.
_DONE_LINE = re.compile(
    re.escape(DONE_MARKER) + r" ([0-9]+)\.([0-9]{2}) ([0-9]+)\.([0-9]{2})(" + re.escape(INCOMPLETE_SUFFIX) + r")?\n"
)

# Run as `python3 -I -S -c READONLY_PROGRAM`: mount_setattr(AT_FDCWD, "/", AT_RECURSIVE, {attr_set=MOUNT_ATTR_RDONLY})
# through syscall 442 (x86_64 and the generic table); a failure exits nonzero, which stops the setup.
READONLY_PROGRAM = """import ctypes, sys
libc = ctypes.CDLL(None, use_errno=True)
attr = (ctypes.c_uint64 * 4)(1, 0, 0, 0)
if libc.syscall(442, -100, b"/", 0x8000, attr, 32) != 0:
    sys.exit("mount_setattr: errno %d" % ctypes.get_errno())
"""

# Run as `python3 -I -S -c CONFINE_PROGRAM <argv...>` right before the program's timeout: it starts a new session,
# joins a new anonymous session keyring (keyctl KEYCTL_JOIN_SESSION_KEYRING, syscall 250; ENOSYS means the kernel
# has no keyrings), installs FILTER with prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER) under the no_new_privs flag
# setpriv set, prints READY_MARKER on stderr, and execs argv. FILTER, for x86_64: any other ABI is killed; x32
# syscalls, io_uring (425 to 427), the keyring calls add_key, request_key, and keyctl (248 to 250),
# socket(AF_VSOCK), setrlimit(RLIMIT_CORE), and a prlimit64 that sets RLIMIT_CORE or any limit of another
# process fail with EPERM; everything else is allowed. A failure exits
# nonzero with a "lassi-sandbox confine:" line, before the ready line when it comes before the exec, and argv
# never runs. Importing it under another __name__ only defines FILTER and confine (the local tests do that).
CONFINE_PROGRAM = """from __future__ import annotations
import ctypes, errno, os, struct, sys
NR, ARCH, ARG0, ARG1, ARG2, ARG2_HIGH = 0, 4, 16, 24, 32, 36
LOAD, JEQ, JGE, RET = 0x20, 0x15, 0x35, 0x06
ALLOW, EPERM, KILL = 0x7FFF0000, 0x00050001, 0x80000000
# Row n is (code, jt, jf, k); a jump goes to row n + 1 + jt when its test holds, else to row n + 1 + jf.
FILTER = [
    (LOAD, 0, 0, ARCH),  # 0: load the audit architecture
    (JEQ, 1, 0, 0xC000003E),  # 1: x86_64 -> 3, else 2
    (RET, 0, 0, KILL),  # 2: any other ABI: kill the process
    (LOAD, 0, 0, NR),  # 3: load the syscall number
    (JGE, 20, 0, 0x40000000),  # 4: an x32 syscall -> 25 EPERM, else 5
    (JEQ, 6, 0, 160),  # 5: setrlimit -> 12, else 6
    (JEQ, 7, 0, 302),  # 6: prlimit64 -> 14, else 7
    (JEQ, 14, 0, 41),  # 7: socket -> 22, else 8
    (JGE, 15, 0, 428),  # 8: 428 and above -> 24 ALLOW, else 9
    (JGE, 15, 0, 425),  # 9: io_uring (425 to 427) -> 25 EPERM, else 10
    (JGE, 13, 0, 251),  # 10: 251 to 424 -> 24 ALLOW, else 11
    (JGE, 13, 12, 248),  # 11: add_key, request_key, keyctl (248 to 250) -> 25 EPERM, else 24 ALLOW
    (LOAD, 0, 0, ARG0),  # 12: setrlimit: load the resource
    (JEQ, 11, 10, 4),  # 13: RLIMIT_CORE -> 25 EPERM, else 24 ALLOW
    (LOAD, 0, 0, ARG2),  # 14: prlimit64: load the low half of new_limit
    (JEQ, 0, 2, 0),  # 15: low half 0 -> 16, else 18
    (LOAD, 0, 0, ARG2_HIGH),  # 16: load the high half of new_limit
    (JEQ, 6, 0, 0),  # 17: new_limit NULL, a read -> 24 ALLOW, else 18
    (LOAD, 0, 0, ARG0),  # 18: load the pid
    (JEQ, 0, 5, 0),  # 19: pid 0, itself -> 20, else 25 EPERM
    (LOAD, 0, 0, ARG1),  # 20: load the resource
    (JEQ, 3, 2, 4),  # 21: RLIMIT_CORE -> 25 EPERM, else 24 ALLOW
    (LOAD, 0, 0, ARG0),  # 22: socket: load the address family
    (JEQ, 1, 0, 40),  # 23: AF_VSOCK -> 25 EPERM, else 24 ALLOW
    (RET, 0, 0, ALLOW),  # 24: allow the call
    (RET, 0, 0, EPERM),  # 25: fail the call with EPERM
]
def confine() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    os.setsid()
    if libc.syscall(250, 1, None) < 0 and ctypes.get_errno() != errno.ENOSYS:
        raise OSError(ctypes.get_errno(), "keyctl KEYCTL_JOIN_SESSION_KEYRING failed")
    code = b"".join(struct.pack("=HBBI", *rule) for rule in FILTER)
    rules = ctypes.create_string_buffer(code, len(code))
    program = ctypes.create_string_buffer(struct.pack("=HxxxxxxQ", len(FILTER), ctypes.addressof(rules)), 16)
    if libc.prctl(22, 2, program, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl PR_SET_SECCOMP failed")
    os.write(2, b"@READY_MARKER@\\n")
    os.execvp(sys.argv[1], sys.argv[1:])
if __name__ == "__main__":
    try:
        confine()
    except OSError as exc:
        sys.exit("lassi-sandbox confine: %s" % exc)
""".replace("@READY_MARKER@", READY_MARKER)

# Run as `python3 -I -S -c COPY_BACK_PROGRAM <upper> <target> <cap bytes>` once the program's namespace has ended.
# It lists the upper layer's directories and regular files first, leaving out entries more than COPY_DEPTH
# levels deep (path components) or with more than COPY_PATH bytes of relative path. If the listed files total
# more than the cap (each hard link counted, a sparse file at its full size), it copies nothing; otherwise it
# copies them into the target, with their permission bits (owner read-write added, group and other write
# removed), reading at most the listed size of each. Each file is written to a new file under a temporary name
# (.lassi-copy-back.<pid>.<n>, created exclusively) in its target directory and renamed over the target, so a
# host file is never opened for writing and a hard link it has to a file outside is never written through. When
# it left anything out it prints one "lassi-sandbox copy-back:" line and exits 3.
# Symbolic links and special files (FIFOs, devices, overlay whiteouts) are skipped, a symbolic link in the target
# is never followed or replaced, an entry whose target has another type is skipped with everything under it, and
# nothing else in the target changes. Any other failure exits 1 with a "copy-back failed:" line, after removing
# the temporary file it was writing. Silent on success.
COPY_BACK_PROGRAM = """from __future__ import annotations
import os, stat, sys
MAX_DEPTH, MAX_PATH, BLOCK = @COPY_DEPTH@, @COPY_PATH@, 1048576
FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
def kind(path: str) -> int | None:
    try:
        return stat.S_IFMT(os.lstat(path).st_mode)
    except FileNotFoundError:
        return None
def listing(upper: str) -> tuple[list[tuple[str, os.stat_result]], list[str]]:
    entries, skipped, stack = [], [], [("", 0)]
    while stack:
        relative, depth = stack.pop()
        with os.scandir(os.path.join(upper, relative)) as found:
            children = sorted((entry.name, entry.stat(follow_symlinks=False)) for entry in found)
        for name, info in children:
            path = os.path.join(relative, name)
            if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
                continue
            if depth >= MAX_DEPTH or len(os.fsencode(path)) > MAX_PATH:
                skipped.append(path)
                continue
            entries.append((path, info))
            if stat.S_ISDIR(info.st_mode):
                stack.append((path, depth + 1))
    return entries, skipped
def create(directory: str, mode: int) -> tuple[int, str]:
    for number in range(1000):
        path = os.path.join(directory, ".lassi-copy-back.%d.%d" % (os.getpid(), number))
        try:
            return os.open(path, FLAGS, mode), path
        except FileExistsError:
            continue
    raise FileExistsError("no free temporary name in " + directory)
def copy_file(source: str, target: str, info: os.stat_result) -> None:
    mode = (stat.S_IMODE(info.st_mode) & 0o755) | 0o600
    handle, path = create(os.path.dirname(target), mode)
    try:
        with os.fdopen(handle, "wb") as writer, open(source, "rb") as reader:
            if hasattr(os, "fchmod"):
                os.fchmod(writer.fileno(), mode)
            left = info.st_size
            while left > 0:
                block = reader.read(min(left, BLOCK))
                if not block:
                    break
                writer.write(block)
                left -= len(block)
        os.replace(path, target)
    except OSError:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise
def copy(upper: str, target: str, entries: list[tuple[str, os.stat_result]]) -> None:
    blocked = set()
    for path, info in entries:
        if os.path.dirname(path) in blocked:
            blocked.add(path)
            continue
        dest = os.path.join(target, path)
        existing = kind(dest)
        if stat.S_ISDIR(info.st_mode):
            if existing is None:
                os.mkdir(dest, 0o755)
            elif existing != stat.S_IFDIR:
                blocked.add(path)
        elif existing in (None, stat.S_IFREG):
            copy_file(os.path.join(upper, path), dest, info)
def main(upper: str, target: str, cap: int) -> int:
    entries, skipped = listing(upper)
    total = sum(info.st_size for _path, info in entries if stat.S_ISREG(info.st_mode))
    if total > cap:
        note = "the new files total %d bytes, over the %d-byte workdir cap; none was copied back" % (total, cap)
        print("lassi-sandbox copy-back: " + note, file=sys.stderr)
        return 3
    copy(upper, target, entries)
    if skipped:
        note = "left out %d entries more than %d levels deep or %d bytes of path, the first %r" % (
            len(skipped), MAX_DEPTH, MAX_PATH, skipped[0][:200])
        print("lassi-sandbox copy-back: " + note, file=sys.stderr)
        return 3
    return 0
try:
    status = main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
except OSError as exc:
    sys.exit("copy-back failed: %s" % exc)
sys.exit(status)
""".replace("@COPY_DEPTH@", str(COPY_DEPTH)).replace("@COPY_PATH@", str(COPY_PATH))

# Run as `awk MOUNT_CHECK /proc/self/mountinfo`: prints "writable host mount: <point> <fstype> <source>" on stderr
# for each mount whose per-mount options start with rw, unless it is one of the setup's own (source lassi-*, type
# tmpfs, devpts, or overlay), and then exits 1. Silent on success. It trusts the setup's mounts by name and type,
# not by mount point (see Known limits in the module docstring).
MOUNT_CHECK = """{
  fs = ""
  src = ""
  for (i = 7; i <= NF; i++) {
    if ($i == "-") {
      j = i + 1
      k = i + 2
      fs = $j
      src = $k
      break
    }
  }
  split($6, options, ",")
  if (options[1] == "rw" && !(src ~ /^lassi-/ && (fs == "tmpfs" || fs == "devpts" || fs == "overlay"))) {
    print "writable host mount: " $5 " " fs " " src > "/dev/stderr"
    bad = 1
  }
}
END {
  exit bad
}"""

# Positional layout: workdir, harness or "", toolchains root or "", disk cap in bytes, N, the N hidden roots, cpu
# seconds, wall seconds, kill-after seconds, then, only when the spec sets the program's environment,
# ENVIRONMENT_MARKER and one NAME=value element per variable, then the program argv. The marker makes the
# program's argv `env -i -- NAME=value... <argv>`, so env, found on SANDBOX_PATH like every other tool the
# chain runs, gives the program exactly those variables. The steps and their order are listed in the
# module docstring. The script stays one constant text, not a set of smaller scripts, because it must be a single
# `sh -c` argument that takes every path and number as a positional parameter: a second script would have to be
# found on a path or passed through the first one's text. The staging tmpfs on /tmp holds: lower (the read-only
# workdir bind), harness, toolchains, layer (the capped tmpfs with upper and work), workdir (the overlay), dev,
# and target (the copy-back bind). The root loops count with a string's length, since the script holds no
# command or arithmetic substitution, and the clock readings for the done line come from the read builtin; the
# private /tmp covers the staging tmpfs until the copy-back unmounts it.
_SETUP_TEMPLATE = r"""set -eu
for tool in env nice setpriv prlimit timeout python3 awk unshare; do
  command -v "$tool" > /dev/null
done
workdir=$1
harness=$2
toolchains=$3
disk=$4
count=$5
shift 5
for number in "$disk" "$count"; do
  case $number in
    '' | 0* | *[!0-9]*) exit 2 ;;
  esac
done
echo 0 > /proc/sys/user/max_user_namespaces
echo 0 > /proc/self/coredump_filter
python3 -I -S -c '@READONLY_PROGRAM@'
mount -t tmpfs -o size=1m,nr_inodes=64,mode=0700 lassi-stage /tmp
mkdir /tmp/lower /tmp/harness /tmp/toolchains /tmp/layer /tmp/workdir /tmp/dev /tmp/target
mount --bind "$workdir" /tmp/lower
if [ -n "$harness" ]; then
  mount --bind "$harness" /tmp/harness
fi
if [ -n "$toolchains" ]; then
  mount --rbind "$toolchains" /tmp/toolchains
fi
mount -t tmpfs -o "size=${disk},nr_inodes=4096,mode=0755" lassi-wd /tmp/layer
mkdir /tmp/layer/upper /tmp/layer/work
mount -t overlay -o userxattr,lowerdir=/tmp/lower,upperdir=/tmp/layer/upper,workdir=/tmp/layer/work lassi-workdir \
  /tmp/workdir
mount -t tmpfs -o size=64k,nr_inodes=64,mode=0755,nosuid,noexec lassi-dev /tmp/dev
for node in null zero full random urandom tty; do
  touch "/tmp/dev/$node"
  mount --bind "/dev/$node" "/tmp/dev/$node"
done
mkdir /tmp/dev/pts /tmp/dev/shm
mount -t devpts -o newinstance,ptmxmode=0666,mode=0620 lassi-devpts /tmp/dev/pts
mount -t tmpfs -o size=64m,mode=1777,nosuid,nodev lassi-shm /tmp/dev/shm
ln -s /proc/self/fd /tmp/dev/fd
ln -s /proc/self/fd/0 /tmp/dev/stdin
ln -s /proc/self/fd/1 /tmp/dev/stdout
ln -s /proc/self/fd/2 /tmp/dev/stderr
ln -s pts/ptmx /tmp/dev/ptmx
mount -o remount,bind,ro,nosuid,noexec /tmp/dev
mount --move /tmp/dev /dev
hidden=
for root in "$@"; do
  [ "${#hidden}" -lt "$count" ] || break
  mount -t tmpfs -o size=1m,nr_inodes=1024,mode=0755 lassi-hide "$root"
  hidden="${hidden}x"
done
mkdir -p "$workdir"
if [ -n "$harness" ]; then
  mkdir -p "$harness"
fi
if [ -n "$toolchains" ]; then
  mkdir -p "$toolchains"
fi
mount --move /tmp/workdir "$workdir"
if [ -n "$harness" ]; then
  mount --move /tmp/harness "$harness"
fi
if [ -n "$toolchains" ]; then
  mount --move /tmp/toolchains "$toolchains"
fi
hidden=
for root in "$@"; do
  [ "${#hidden}" -lt "$count" ] || break
  mount -o remount,bind,ro "$root"
  hidden="${hidden}x"
done
shift "$count"
for dir in /sys/class /sys/bus /sys/devices/* /var/*; do
  case $dir in
    /sys/devices/system | /var/tmp) ;;
    *)
      if [ -d "$dir" ] && [ ! -L "$dir" ]; then
        mount -t tmpfs -o ro,size=4k,nr_inodes=8,mode=0555 lassi-sys "$dir"
      fi
      ;;
  esac
done
for dir in /var/tmp /run; do
  if [ -d "$dir" ]; then
    mount -t tmpfs -o size=64m,mode=1777 lassi-private "$dir"
  fi
done
mount -t tmpfs -o size=64m,mode=1777 lassi-tmp /tmp
awk '@MOUNT_CHECK@' /proc/self/mountinfo
cd "$workdir"
cpu=$1
wall=$2
kill_after=$3
shift 3
if [ "$1" = @ENVIRONMENT_MARKER@ ]; then
  shift
  set -- env -i -- "$@"
fi
status=0
read -r started rest < /proc/uptime
unshare --pid --ipc --fork --kill-child -- env -i PATH="$PATH" HOME="$workdir" LANG=C.UTF-8 TMPDIR=/tmp \
  nice -n 19 \
  setpriv --no-new-privs --inh-caps=-all --bounding-set=-all -- \
  prlimit --cpu="$cpu" --core=1 --fsize="$disk" -- \
  python3 -I -S -c '@CONFINE_PROGRAM@' \
  timeout --kill-after="$kill_after" "$wall" "$@" || status=$?
trap 'echo "lassi-sandbox: the setup stopped after the program ended" >&2' EXIT
read -r ended rest < /proc/uptime
cd /
umount /tmp
for dir in /var/tmp /run; do
  if [ -d "$dir" ]; then
    umount "$dir"
  fi
done
umount /dev/shm
umount "$workdir"
mount --bind /tmp/lower /tmp/target
mount -o remount,bind,rw /tmp/target
copied=0
python3 -I -S -c '@COPY_BACK_PROGRAM@' /tmp/layer/upper /tmp/target "$disk" || copied=$?
case $copied in
  0) note= ;;
  3) note=' incomplete' ;;
  *) exit "$copied" ;;
esac
trap - EXIT
echo "lassi-sandbox-done $started $ended$note" >&2
exit "$status"
"""


def _embed(template: str, programs: dict[str, str]) -> str:
    """Return `template` with each @NAME@ replaced by its program, which must not hold a single quote."""
    for name, program in programs.items():
        if "'" in program or template.count(f"@{name}@") != 1:
            raise AssertionError(f"{name} cannot be embedded in single quotes exactly once")
        template = template.replace(f"@{name}@", program)
    return template


SETUP_SCRIPT = _embed(
    _SETUP_TEMPLATE,
    {
        "READONLY_PROGRAM": READONLY_PROGRAM,
        "MOUNT_CHECK": MOUNT_CHECK,
        "CONFINE_PROGRAM": CONFINE_PROGRAM,
        "COPY_BACK_PROGRAM": COPY_BACK_PROGRAM,
        "ENVIRONMENT_MARKER": ENVIRONMENT_MARKER,
    },
)


class SandboxUnavailableError(RuntimeError):
    """The sandbox cannot run the program, or could not finish around it.

    A tool or the workdir is missing, setup failed or found a writable host
    mount, no root is hidden, or the setup ran the program but did not finish
    the copy-back. The program did not run, or its result is not reported,
    and nothing is ever run unsandboxed instead.
    """


def _dir_holding(path: PurePath, directories: Sequence[str]) -> str | None:
    """Return the entry of `directories` that `path` is or lies under, compared as normalized POSIX text, or None.

    The comparison is lexical: `..` segments are resolved, symbolic links are
    not followed.
    """
    text = posixpath.normpath(path.as_posix())
    if text.startswith("//"):
        text = "/" + text.lstrip("/")
    posix = PurePosixPath(text)
    for name in directories:
        directory = PurePosixPath(name)
        if posix == directory or directory in posix.parents:
            return name
    return None


def _check_mount_path(what: str, path: PurePath, *, exposed: bool) -> None:
    """Raise ValueError unless `path` is absolute and outside every directory SETUP_SCRIPT covers with tmpfs.

    An `exposed` path (the workdir, the harness, or the toolchains root) must
    also lie outside SYSTEM_DIRS, whose entries are hidden inside.
    """
    if not path.is_absolute():
        raise ValueError(f"the sandbox {what} must be an absolute path, got {str(path)!r}")
    private = _dir_holding(path, PRIVATE_DIRS)
    if private is not None:
        raise ValueError(f"the sandbox {what} {str(path)!r} lies under {private}, which a private tmpfs hides inside")
    system = _dir_holding(path, SYSTEM_DIRS) if exposed else None
    if system is not None:
        raise ValueError(f"the sandbox {what} {str(path)!r} lies under {system}, whose entries a tmpfs hides inside")


def _covers(outer: PurePath, inner: PurePath) -> bool:
    """Return True when `outer` is `inner` or one of its parents."""
    return outer == inner or outer in inner.parents


def _outermost(roots: Sequence[PurePath]) -> tuple[PurePath, ...]:
    """Return `roots` without duplicates and without roots under another root, in first-seen order."""
    unique = list(dict.fromkeys(roots))
    return tuple(root for root in unique if not any(other in root.parents for other in unique))


def _check_int(name: str, value: object) -> None:
    """Raise ValueError unless `value` is an int >= 1 (bool refused)."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be an integer >= 1, got {value!r}")


def _checked_environment(environment: object) -> dict[str, str] | None:
    """Return a copy of a program environment after checking it, or None for None (the program defaults).

    Raise ValueError, naming the variable, unless `environment` is a mapping
    (not a string) whose names are in ENVIRONMENT_NAMES and whose values are
    strings without NUL.
    """
    if environment is None:
        return None
    if isinstance(environment, (str, bytes)) or not isinstance(environment, Mapping):
        raise ValueError(f"a sandbox environment must be a mapping of names to values, got {environment!r}")
    checked: dict[str, str] = {}
    for name, value in environment.items():
        if not isinstance(name, str) or name not in ENVIRONMENT_NAMES:
            allowed = ", ".join(sorted(ENVIRONMENT_NAMES))
            raise ValueError(f"the sandbox environment may not set {name!r}; allowed names: {allowed}")
        if not isinstance(value, str) or "\0" in value:
            raise ValueError(f"the sandbox environment's {name} must be a string without NUL, got {value!r}")
        checked[name] = value
    return checked


@dataclass(frozen=True)
class SandboxSpec:
    """Where a sandboxed run may write, what it sees, and its disk cap.

    `workdir` is the per-trial build directory, the only writable host
    directory; the program writes to an overlay of it capped at `disk_mb`
    MiB (at most half the memory limit, workdir_cap_bytes), and its new files
    are copied back afterwards. `hidden_roots` are host directories hidden
    behind tmpfs inside (for example the scratch root, $HOME, and the runs
    root), except for the path skeleton to the workdir, the harness, and the
    toolchains root; at least one is required, and duplicates and roots
    under another root are dropped (first-seen order kept). `harness` and
    `toolchains`, when set, are bind-mounted read-only at their own paths.
    `tasks_max` caps the tasks in the run's cgroup. `environment` is the
    program's environment: None keeps the program defaults (PATH=
    SANDBOX_PATH, HOME=<workdir>, LANG=C.UTF-8, TMPDIR=/tmp) and the P0.16
    command; a mapping gives the program exactly those variables, with
    names from ENVIRONMENT_NAMES and string values without NUL, and is kept
    as a read-only copy. ValueError is raised
    when a path is relative or lies under one of PRIVATE_DIRS (the private
    tmpfs would hide it), when the workdir, the harness, or the toolchains
    root lies under one of SYSTEM_DIRS, when a hidden root is a filesystem
    root, when the workdir is or contains a hidden root, when the harness or
    the toolchains root is, contains, or lies under the workdir, or is or
    contains a hidden root, when tasks_max or disk_mb is not an integer
    >= 1, or when the environment is not such a mapping. The checks are
    lexical; callers resolve symbolic links first
    (lassi.executors.native does). Paths are stored as Path and the roots as
    a tuple.
    """

    workdir: Path
    hidden_roots: tuple[Path, ...]
    harness: Path | None = None
    toolchains: Path | None = None
    tasks_max: int = 256
    disk_mb: int = WORKDIR_DISK_MB
    environment: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        """Normalize the paths to Path, copy the environment, and check every field; ValueError on the first problem."""
        if isinstance(self.hidden_roots, (str, os.PathLike)):
            raise ValueError(f"hidden_roots must be a sequence of paths, got {self.hidden_roots!r}")
        roots = tuple(Path(root) for root in self.hidden_roots)
        object.__setattr__(self, "workdir", Path(self.workdir))
        for name in ("harness", "toolchains"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, Path(getattr(self, name)))
        if not roots:
            raise ValueError("a sandbox needs at least one hidden root")
        exposures = [(name, getattr(self, name)) for name in ("harness", "toolchains") if getattr(self, name)]
        _check_mount_path("workdir", self.workdir, exposed=True)
        for what, path in [*(("hidden root", root) for root in roots), *exposures]:
            _check_mount_path(what, path, exposed=what != "hidden root")
        if any(root.parent == root for root in roots):
            raise ValueError("a hidden root cannot be a filesystem root, which a tmpfs on it would not hide")
        if any(_covers(self.workdir, root) for root in roots):
            raise ValueError(f"the workdir {str(self.workdir)!r} cannot be or contain a hidden root")
        for what, path in exposures:
            if _covers(path, self.workdir) or _covers(self.workdir, path) or any(_covers(path, r) for r in roots):
                raise ValueError(
                    f"the {what} {str(path)!r} cannot be, contain, or lie under the workdir, or be or contain a "
                    "hidden root"
                )
        object.__setattr__(self, "hidden_roots", _outermost(roots))
        _check_int("tasks_max", self.tasks_max)
        _check_int("disk_mb", self.disk_mb)
        environment = _checked_environment(self.environment)
        object.__setattr__(self, "environment", None if environment is None else MappingProxyType(environment))


@dataclass(frozen=True)
class SandboxResult:
    """How a sandboxed run ended.

    `returncode` is in the shell's form (a death by signal N is 128 + N; -1
    is the runner's own timeout). `stderr` is the program's, without the
    setup script's ready and done lines; when `workdir_incomplete` is set,
    the copy-back's "lassi-sandbox copy-back:" line follows it. `wall_s` is
    the elapsed time of the whole sandbox command, setup and copy-back
    included. `hang` means the program reached the wall limit and ended by a
    timeout or a kill; `killed` means a SIGKILL death before the wall limit,
    which may be a memory or CPU-time limit kill or the program killing
    itself (see classify). `stdout_truncated` and `stderr_truncated` are the
    runner's flags: that stream passed the output cap and only its head and
    tail are kept. `workdir_incomplete` means the copy-back left part of what
    the program wrote in the workdir out (all of it when the new files total
    more than the disk cap; the entries past COPY_DEPTH or COPY_PATH
    otherwise), or did not finish because the whole sandbox was killed (the
    runner's own timeout, the backstop, or a memory kill of the setup), so
    the host workdir may lack any of it. `program_s` is the program's own
    time: the difference of the two /proc/uptime readings the setup took
    right before and right after the program's namespace, from its done
    line, in 10 ms steps (each reading is cut to a step, so the time itself
    may be up to 10 ms longer or shorter); it covers the program's chain
    (env to the innermost timeout) and not the setup or the copy-back. It
    is None when no done line counts (the whole sandbox was killed).
    """

    returncode: int
    stdout: str
    stderr: str
    wall_s: float
    hang: bool
    killed: bool
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    workdir_incomplete: bool = False
    program_s: float | None = None


def _check_request(argv: Sequence[str], limits: Limits, environment: Mapping[str, str] | None = None) -> None:
    """Raise ValueError unless argv is a non-empty sequence of strings and the limits can be enforced.

    The program may not be named ENVIRONMENT_MARKER, which SETUP_SCRIPT
    reads as the start of the program's environment. With an `environment`,
    the program runs as `env -i -- NAME=value... <argv>`, so argv[0] may not
    hold "=" (env would read it as a variable) or be "-".
    """
    if isinstance(argv, str) or not argv:
        raise ValueError(f"argv must be a non-empty sequence of strings, got {argv!r}")
    if not all(isinstance(part, str) for part in argv):
        raise ValueError(f"every argv element must be a string, got {list(argv)!r}")
    if argv[0] == ENVIRONMENT_MARKER:
        raise ValueError(f"the program may not be named {ENVIRONMENT_MARKER!r}, the sandbox's environment marker")
    if environment is not None and ("=" in argv[0] or argv[0] == "-"):
        raise ValueError(f"with an environment, the program's name may not hold '=' or be '-', got {argv[0]!r}")
    if not (isinstance(limits.wall_s, (int, float)) and math.isfinite(limits.wall_s) and limits.wall_s > 0):
        raise ValueError(f"wall_s must be a finite number > 0, got {limits.wall_s!r}")
    for name in ("memory_mb", "cpus"):
        value = getattr(limits, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be an integer >= 1, got {value!r}")


def _wall_seconds(limits: Limits) -> int:
    """Return the whole-second wall limit the innermost timeout enforces: wall_s rounded up, at least 1."""
    return max(1, math.ceil(limits.wall_s))


def workdir_cap_bytes(spec: SandboxSpec, limits: Limits) -> int:
    """Return the workdir disk cap a run gets, in bytes: spec.disk_mb MiB, but at most half of limits.memory_mb.

    tmpfs pages count toward the run's memory limit (plans/spikes/
    p0-sandbox-hardening.md, probe E), so a cap at half of it makes a write
    past the cap fail with ENOSPC rather than a memory kill, and leaves the
    other half for the program and the copy-back. The cap is at least 1 MiB.
    It is also the program's file size limit (prlimit --fsize).
    """
    return min(spec.disk_mb, max(1, limits.memory_mb // 2)) << 20


def _passed_variables() -> list[str]:
    """Return NAME=value for each of PASSED_VARIABLES that is set and not empty in this process's environment."""
    return [f"{name}={os.environ[name]}" for name in PASSED_VARIABLES if os.environ.get(name)]


def sandbox_command(spec: SandboxSpec, argv: Sequence[str], limits: Limits) -> list[str]:
    """Return the command that runs `argv` in the sandbox described by `spec` under `limits`.

    The wall limit is wall_s rounded up to whole seconds (at least 1). The
    CPU-time budget is wall_s x cpus, rounded up (at least 1): CPU is capped
    as CPU time because the cgroup cpu controller is not delegated (OQ-011).
    The backstop RuntimeMaxSec is the wall limit plus KILL_AFTER_S plus
    OUTER_MARGIN_S. The disk cap is workdir_cap_bytes. env -i gives the
    sandbox PATH=SANDBOX_PATH and, from this process's environment when set,
    PASSED_VARIABLES, read when called. Every path and argument is its own
    element after SETUP_SCRIPT (read when called), in the script's
    positional layout. When spec.environment is set, ENVIRONMENT_MARKER and
    one NAME=value element per variable, in name order, come right before
    argv, which stays the command's tail; with None the command is the
    P0.16 command. Raise ValueError when argv is empty or names the program
    as _check_request refuses, or wall_s is not > 0, or memory_mb or cpus
    is below 1.
    """
    _check_request(argv, limits, spec.environment)
    wall = _wall_seconds(limits)
    cpu = max(1, math.ceil(limits.wall_s * limits.cpus))
    optional = ["" if path is None else str(path) for path in (spec.harness, spec.toolchains)]
    roots = [str(root) for root in spec.hidden_roots]
    environment = ["env", "-i", f"PATH={SANDBOX_PATH}", *_passed_variables()]
    scope = ["systemd-run", "--user", "--scope", "--quiet"]
    scope += ["-p", f"MemoryMax={limits.memory_mb}M", "-p", "MemorySwapMax=0", "-p", f"TasksMax={spec.tasks_max}"]
    scope += ["-p", f"RuntimeMaxSec={wall + KILL_AFTER_S + OUTER_MARGIN_S}", "-p", "TimeoutStopSec=1"]
    namespaces = ["unshare", "-rinmpfu", "--mount-proc", "--kill-child"]
    layout = [str(spec.workdir), *optional, str(workdir_cap_bytes(spec, limits)), str(len(roots)), *roots]
    layout += [str(cpu), str(wall), str(KILL_AFTER_S)]
    if spec.environment is not None:
        layout += [ENVIRONMENT_MARKER, *(f"{name}={spec.environment[name]}" for name in sorted(spec.environment))]
    setup = ["sh", "-c", SETUP_SCRIPT, "sh", *layout]
    return ["prlimit", "--core=1", "--", *environment, *scope, *namespaces, *setup, *argv]


def classify(returncode: int, wall_s: float, limit_wall_s: float) -> tuple[bool, bool]:
    """Return (hang, killed) for a sandboxed run's exit status and elapsed wall time.

    `returncode` is in the shell's form, as the spike recorded it with bash:
    a death by signal N is 128 + N, and -1 is the runner's own timeout.
    Sandbox.run converts the runner's -N (Popen's form) before calling this,
    so the backstop's SIGKILL reaching the runner as -9 arrives here as 137.
    The mapping follows the exit statuses in plans/spikes/p0-sandbox.md and
    its addenda. `wall_s` is the elapsed time Sandbox.run measured: the
    program's own time (SandboxResult.program_s) plus one 10 ms step, since
    each clock reading is cut to a step and a timeout that fired must never
    be missed, or the whole command's time when no done line counts (the
    whole sandbox was killed, past the program's end). A hang needs both the
    status and that time, so setup and copy-back time never count, and a
    program cannot pass for a hang by exiting 124, 137, or 143 itself before
    the wall limit, except within the program's own time's margin of it:
    the time its chain takes to start the innermost timeout and to end after
    it (not measured) plus up to 20 ms of clock steps:

    - 124, 137, 143, or -1 with wall_s >= limit_wall_s: hang. The run
      reached the wall limit, so the stop counts as the wall limit's: 124
      is the innermost timeout, 137 its kill after the grace or the scope's
      RuntimeMaxSec backstop (which also gives 143), and -1 the runner's own
      timeout (lassi.toolchains capped_runner).
    - 137 with wall_s < limit_wall_s: killed. A SIGKILL death before the
      wall limit: a memory (cgroup MemoryMax) or CPU-time (prlimit --cpu)
      limit kill, or the program killing itself, which the status cannot
      tell apart.
    - Anything else: (False, False), the program's own exit status. That
      includes 124, 143, and -1 before the wall limit, and a crash (139,
      134).
    """
    if returncode in (124, 137, 143, -1) and wall_s >= limit_wall_s:
        return True, False
    if returncode == 137:
        return False, True
    return False, False


def _shell_status(returncode: int, elapsed_s: float, timeout_s: float) -> int:
    """Return the runner's exit status in the shell's form that classify reads.

    The runner (Popen) reports a death by signal N as -N, where a shell and
    the spike's measurements give 128 + N. -1 stays -1 only when the runner's
    own timeout expired (elapsed_s >= timeout_s); before that it is a death by
    SIGHUP, 129.
    """
    if returncode == -1 and elapsed_s >= timeout_s:
        return -1
    if returncode < 0:
        return 128 - returncode
    return returncode


def _split_ready(stderr: str) -> tuple[bool, str]:
    """Return whether stderr holds the ready line, and stderr with that first line removed.

    CONFINE_PROGRAM prints the line once the program's chain is confined,
    right before it execs the program's timeout. The line counts only at the
    start of a line. Only the setup's and the chain's output can come before
    it, since the program starts after it, so the program cannot forge a
    sandbox that started.
    """
    line = READY_MARKER + "\n"
    if stderr.startswith(line):
        return True, stderr[len(line) :]
    index = stderr.find("\n" + line)
    if index < 0:
        return False, stderr
    return True, stderr[: index + 1] + stderr[index + 1 + len(line) :]


def _split_done(stderr: str) -> tuple[int | None, bool, str]:
    """Return the program's own time in clock steps from SETUP_SCRIPT's done line, its incomplete mark, and the rest.

    The done line is DONE_MARKER, the setup's two /proc/uptime readings
    (right before and right after the program's namespace), and
    INCOMPLETE_SUFFIX when the copy-back left something out. The setup
    prints it last, after the program's namespace has ended, so it follows
    the program's last byte even when that is not a newline, and nothing the
    program prints can come after it: the last marker in stderr is the
    setup's. The time, in UPTIME_STEPS_PER_S steps, is the second reading
    minus the first; it is None, with stderr kept whole, when stderr does
    not end with such a line or the second reading comes before the first.
    """
    index = stderr.rfind(DONE_MARKER)
    match = _DONE_LINE.fullmatch(stderr, index) if index >= 0 else None
    if match is None:
        return None, False, stderr
    started = int(match[1]) * UPTIME_STEPS_PER_S + int(match[2])
    ended = int(match[3]) * UPTIME_STEPS_PER_S + int(match[4])
    if ended < started:
        return None, False, stderr
    return ended - started, match[5] is not None, stderr[:index]


def _setup_failure(result: CommandResult) -> str:
    """Return the SandboxUnavailableError message for a run whose setup or confinement never reached the program."""
    tail = result.stderr.strip()[-_STDERR_TAIL:] or "(no stderr)"
    return (
        f"the sandbox setup or the program's confinement failed (exit status {result.returncode}); "
        f"the program did not run: {tail}"
    )


def _start_failure(exc: OSError, tool: str, workdir: Path) -> str:
    """Return the SandboxUnavailableError message for a sandbox command the runner could not start."""
    if isinstance(exc, FileNotFoundError):
        missing = exc.filename or tool
        return f"the sandbox cannot start: {missing} was not found (a tool it needs, or the workdir); nothing ran"
    where = f" ({exc.filename})" if exc.filename else ""
    reason = exc.strerror or str(exc) or type(exc).__name__
    return f"the sandbox cannot start: running {tool} in the workdir {workdir} failed: {reason}{where}; nothing ran"


def _copy_back_failure(result: CommandResult) -> str:
    """Return the SandboxUnavailableError message for a run whose setup exited without finishing the copy-back."""
    tail = result.stderr.strip()[-_STDERR_TAIL:] or "(no stderr)"
    return (
        f"the sandbox ran the program but did not finish the copy-back of its workdir (exit status "
        f"{result.returncode}); the result is not reported: {tail}"
    )


class Sandbox:
    """Runs a program only inside the sandbox; nothing ever runs unsandboxed."""

    def __init__(self, *, runner: CommandRunner | None = None) -> None:
        """Keep the command runner; None means lassi.toolchains capped_runner, which caps stdout and stderr."""
        self.runner: CommandRunner = capped_runner if runner is None else runner

    def run(self, spec: SandboxSpec, argv: Sequence[str], limits: Limits) -> SandboxResult:
        """Run `argv` in the sandbox described by `spec` under `limits` and classify how it ended.

        The runner gets sandbox_command(spec, argv, limits), the workdir as
        its cwd, and a timeout 10 s past the backstop. Wall time is measured
        with time.monotonic around the runner call. The runner's status is
        converted to the shell's form (_shell_status). The ready line
        (CONFINE_PROGRAM's) and the setup's final done line are removed from
        stderr, and the runner's truncation flags are kept. The done line
        counts only after a normal exit (a status >= 0): the setup exits
        normally right after printing it, so a done line at the end of a
        signal death's stderr is the program's, and stderr is then kept as
        it is. classify compares the program's own time from the done line
        (program_s) plus one clock step, or the whole command's time when no
        done line counts, with the whole-second wall limit the innermost
        timeout enforces. workdir_incomplete is the done line's incomplete
        mark, or True when no done line counts (the copy-back did not
        finish). Validation errors raise ValueError before anything runs.
        SandboxUnavailableError is raised when the runner raises any OSError
        (the command did not start: a missing, unreadable, or unexecutable
        tool or workdir, or a refused cwd, named in the message), when stderr
        lacks the ready line (the setup or the confinement failed before it,
        so the program never ran), and when the ready line is there but the
        done line is not while the runner saw a normal exit (the copy-back
        did not finish); the message quotes the stderr. After a signal death
        or the runner's own timeout, the whole sandbox was killed, and the
        result is returned as classified.
        """
        command = sandbox_command(spec, argv, limits)
        wall = _wall_seconds(limits)
        timeout_s = wall + KILL_AFTER_S + OUTER_MARGIN_S + _RUNNER_MARGIN_S
        start = time.monotonic()
        try:
            result = self.runner(command, spec.workdir, timeout_s)
        except OSError as exc:
            raise SandboxUnavailableError(_start_failure(exc, command[0], spec.workdir)) from exc
        elapsed = time.monotonic() - start
        ready, stderr = _split_ready(result.stderr)
        if not ready:
            raise SandboxUnavailableError(_setup_failure(result))
        exited = result.returncode >= 0
        steps, incomplete, stderr = _split_done(stderr) if exited else (None, False, stderr)
        if exited and steps is None:
            raise SandboxUnavailableError(_copy_back_failure(result))
        returncode = _shell_status(result.returncode, elapsed, timeout_s)
        measured = elapsed if steps is None else (steps + 1) / UPTIME_STEPS_PER_S
        hang, killed = classify(returncode, measured, wall)
        return SandboxResult(
            returncode=returncode,
            stdout=result.stdout,
            stderr=stderr,
            wall_s=elapsed,
            hang=hang,
            killed=killed,
            stdout_truncated=result.stdout_truncated,
            stderr_truncated=result.stderr_truncated,
            workdir_incomplete=incomplete or steps is None,
            program_s=None if steps is None else steps / UPTIME_STEPS_PER_S,
        )


def _timed_out(stderr: str, timeout_s: float) -> str:
    """Return `stderr` ending with a "timed out" line, as the toolchains' runners end a timed-out command's."""
    lines = stderr.rstrip("\n").split("\n")
    if lines[-1].startswith("timed out"):
        return stderr
    if stderr and not stderr.endswith("\n"):
        stderr += "\n"
    return stderr + f"timed out after {timeout_s:g} s\n"


def _compile_stderr(result: SandboxResult, cap_bytes: int | None, timeout_s: float) -> str:
    """Return a compile's stderr ending with a "lassi-sandbox:" line for each thing the sandbox cut or stopped.

    In order: a line for each stream past the output cap `cap_bytes` (None
    when the runner does not name its cap); one when a limit killed the
    compile before its wall limit (SandboxResult.killed: the memory limit or
    the CPU-time limit); one when the whole sandbox was killed before its
    done line (program_s None), so the build dir may lack the compiler's
    outputs; and after a hang the "timed out" line last (_timed_out). A
    compile that ended on its own and within the cap gets none.
    """
    cap = "the compile output cap" if cap_bytes is None else f"the compile output cap of {cap_bytes} bytes"
    cuts = (("stdout", result.stdout_truncated), ("stderr", result.stderr_truncated))
    notes = [
        f"lassi-sandbox: the compiler's {stream} was longer than {cap}; only its head and its last "
        f"{OUTPUT_TAIL_BYTES} bytes are kept"
        for stream, cut in cuts
        if cut
    ]
    if result.killed:
        cpu = max(1, math.ceil(timeout_s * COMPILE_CPUS))
        notes.append(
            f"lassi-sandbox: the compile was killed before its wall limit, probably by its memory limit "
            f"({COMPILE_MEMORY_MB} MiB) or its CPU-time limit ({cpu} s per process), or by a SIGKILL from within"
        )
    if result.program_s is None and not result.hang:
        notes.append(
            "lassi-sandbox: the compile sandbox was killed before it finished, so the build dir may lack the "
            "compiler's outputs"
        )
    stderr = result.stderr
    if notes:
        if stderr and not stderr.endswith("\n"):
            stderr += "\n"
        stderr += "".join(f"{note}\n" for note in notes)
    return _timed_out(stderr, timeout_s) if result.hang else stderr


class SandboxedCompileRunner:
    """A CommandRunner that compiles model-generated sources only inside the sandbox (P0.20).

    Each call runs the compiler command as sandbox_command builds it for
    spec(cwd) and limits(timeout_s), through Sandbox.run, so the P0.16
    mechanisms cover the compile too: prlimit --core=1 (a compiler crash
    stores no core with the host handler, Agent Rule 7), the namespaces, the
    read-only host, and the capped overlay on the build dir. What the
    compiler can read:

    - its build dir (`cwd`, build()'s workdir), the one writable host
      directory, and the pinned toolchains root `toolchains`, read-only;
    - nothing else under the hidden roots, which the stage runner sets to
      $HOME, the scratch root ($LASSI_SCRATCH), and the runs root
      ($LASSI_RUNS_ROOT, and the run's own runs root; those set): only the
      path skeleton to the build dir and the toolchains root shows there, so
      a generated `#include` of an absolute path under them, another trial
      included, finds no file;
    - files elsewhere keep the host user's read permission, as for program
      runs (Known limits in the module docstring): the sandbox hides those
      roots, not the rest of the host.

    The compiler's environment is exactly `environment` (names from
    ENVIRONMENT_NAMES; HOME and TMPDIR refused) plus TMPDIR, its private
    directory COMPILE_TMPDIR under the build dir, which the call creates
    fresh (mode 0700) before the compile and removes afterwards. So its
    temporary files stay in its own view, count toward the workdir disk
    cap, and never land in the scratch root's shared temporary directory.

    Limits: the wall limit is the toolchain's timeout (600 s by default);
    memory, disk, and CPU time are COMPILE_MEMORY_MB, COMPILE_DISK_MB, and
    COMPILE_CPUS, set with wide margin over what the fixture scenarios and
    one benchmark app needed on alpha01 (exploratory, see the constants).
    The setup adds about 0.3 s to each compile there (same run).

    Output: the default inner runner is CappedRunner(COMPILE_OUTPUT_CAP_BYTES),
    so each of stdout and stderr passes through whole up to that cap, far
    above any compile's output (the P0.15 contract keeps compile.stderr
    whole; OUTPUT_CAP_BYTES is for generated programs only), while a
    generated source that makes the compiler print without end fills neither
    the host's memory (the runner's buffers lie outside the sandbox's memory
    limit) nor the disk under compile.stderr. Nothing is cut or stopped
    silently: a "lassi-sandbox:" line ends stderr for each stream past the
    cap (its head and its last OUTPUT_TAIL_BYTES bytes are kept), for a
    compile a limit killed before its wall limit (status 137: the memory or
    the CPU-time limit), and for one whose whole sandbox was killed (the
    build dir may lack its outputs). The status is the compiler's in the
    shell's form (a crash by SIGSEGV is 139). A compile that reaches the wall
    limit returns -1 with stderr ending in a "timed out" line, as the
    toolchains' runners do, so build() reports it as a timeout.
    SandboxUnavailableError propagates, so nothing ever compiles unsandboxed
    instead. env's own statuses (125, 126, 127) come back as the compiler's
    (module docstring); the stage runner's --version check runs through this
    same runner before the first build, so a compiler missing from the view
    stops the run before any build.
    """

    def __init__(
        self,
        *,
        environment: Mapping[str, str],
        toolchains: Path,
        hidden_roots: Sequence[Path],
        runner: CommandRunner | None = None,
    ) -> None:
        """Keep a copy of the environment, the toolchains root, the hidden roots, and the inner runner.

        `runner` None means CappedRunner(COMPILE_OUTPUT_CAP_BYTES), read
        when the runner is made. Raise ValueError when the environment holds
        a name outside ENVIRONMENT_NAMES, a value that is not a string
        without NUL, or TMPDIR (each compile gets its own), or when no hidden
        root is given.
        """
        checked = _checked_environment(environment)
        if checked is None or "TMPDIR" in checked:
            raise ValueError("a compile environment is a mapping without TMPDIR; each compile gets its own TMPDIR")
        if isinstance(hidden_roots, (str, os.PathLike)) or not hidden_roots:
            raise ValueError(f"a compile needs a sequence of hidden roots, got {hidden_roots!r}")
        self.environment: dict[str, str] = checked
        self.toolchains = Path(toolchains)
        self.hidden_roots: tuple[Path, ...] = tuple(Path(root) for root in hidden_roots)
        self.runner: CommandRunner = CappedRunner(COMPILE_OUTPUT_CAP_BYTES) if runner is None else runner

    def __repr__(self) -> str:
        """Show the class, the environment's variable names (never their values), and the roots."""
        return (
            f"{type(self).__name__}(names={sorted(self.environment)!r}, toolchains={str(self.toolchains)!r}, "
            f"hidden_roots={[str(root) for root in self.hidden_roots]!r})"
        )

    def spec(self, workdir: Path) -> SandboxSpec:
        """Return the SandboxSpec of a compile in the build dir `workdir`; it depends on `workdir` alone.

        The workdir, the toolchains root, and the hidden roots are resolved
        (symbolic links followed), so a link cannot carry a hidden root into
        view. There is no harness, the disk cap is COMPILE_DISK_MB, and the
        environment is the compile environment plus TMPDIR=<build
        dir>/COMPILE_TMPDIR. SandboxSpec raises ValueError for a layout it
        refuses.
        """
        build = Path(workdir).resolve()
        return SandboxSpec(
            workdir=build,
            hidden_roots=tuple(root.resolve() for root in self.hidden_roots),
            toolchains=self.toolchains.resolve(),
            disk_mb=COMPILE_DISK_MB,
            environment={**self.environment, "TMPDIR": str(build / COMPILE_TMPDIR)},
        )

    def limits(self, timeout_s: float) -> Limits:
        """Return a compile's Limits: the toolchain's timeout as wall time, COMPILE_MEMORY_MB, and COMPILE_CPUS."""
        return Limits(wall_s=timeout_s, memory_mb=COMPILE_MEMORY_MB, cpus=COMPILE_CPUS)

    def __call__(self, argv: Sequence[str], cwd: Path, timeout_s: float) -> CommandResult:
        """Compile `argv` in the build dir `cwd` inside the sandbox; return the compiler's status and output.

        The output is whole up to the runner's cap, and stderr ends with the
        sandbox's own lines as _compile_stderr adds them. Raise
        SandboxUnavailableError when the private TMPDIR cannot be created or
        the sandbox cannot run the compile, and ValueError for a spec or
        request the sandbox refuses.
        """
        spec = self.spec(cwd)
        tmpdir = spec.workdir / COMPILE_TMPDIR
        shutil.rmtree(tmpdir, ignore_errors=True)
        try:
            tmpdir.mkdir(mode=0o700)
        except OSError as exc:
            raise SandboxUnavailableError(f"cannot create the compile's private TMPDIR {tmpdir}: {exc}") from exc
        try:
            result = Sandbox(runner=self.runner).run(spec, argv, self.limits(timeout_s))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
        return CommandResult(
            -1 if result.hang else result.returncode,
            result.stdout,
            _compile_stderr(result, getattr(self.runner, "cap_bytes", None), timeout_s),
            stdout_truncated=result.stdout_truncated,
            stderr_truncated=result.stderr_truncated,
        )
