"""The command runners and the build steps shared by the compiler toolchains.

lassi.toolchains re-exports CommandResult, CommandRunner, subprocess_runner,
capped_runner, CappedRunner, EnvRunner, and STDERR_ATTACHMENT from here. The
adapters in lassi.toolchains.nvcc, lassi.toolchains.nvcpp,
lassi.toolchains.gcc, and lassi.toolchains.ttmetal_build subclass
CompilerToolchain, which holds build().

Every runner drains stdout and stderr on one thread per pipe while the
command runs, so the command never blocks on a full pipe (threads rather than
selectors, because selectors do not work on pipes on Windows, where the local
tests run). subprocess_runner and EnvRunner keep all of the output, as they
did before P0.16. capped_runner, the sandbox's default runner for programs,
keeps at most OUTPUT_CAP_BYTES of each stream (P0.16 R4; the design is in
plans/spikes/p0-sandbox-hardening.md, probe K and design item 5): the first
OUTPUT_CAP_BYTES - OUTPUT_TAIL_BYTES bytes and the last OUTPUT_TAIL_BYTES
bytes, joined with no marker. CappedRunner does the same at a cap of its
own: the compile sandbox (lassi.executors.sandbox.SandboxedCompileRunner,
P0.20) runs every compile through one whose cap lies far above any
compiler output seen, so the raw compiler stderr attachment stays whole
while a compiler that prints without end cannot fill the host's memory; the
compile runner then says in stderr that the stream was cut. Past a cap the
rest is read and dropped, and CommandResult.stdout_truncated or
stderr_truncated says so. The tail keeps a command's final lines, such as the sandbox setup's done line,
when the output before them was cut. After a timeout, stderr also ends with
the runner's own "timed out" line, which comes on top of the cap. Once a
runner has taken its result, a pipe that a process outside the killed tree
still holds is drained and dropped, so nothing grows after the call returns.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePath, PurePosixPath
from typing import IO

from lassi.core.files import _check_path
from lassi.core.interfaces import BuildResult
from lassi.core.record import Diagnostic
from lassi.toolchains._stderr import compile_diagnostic

# The file, inside the workdir, where build() keeps the compiler's raw stderr.
STDERR_ATTACHMENT = "compile.stderr"
# The artifact file name, inside the workdir.
OUTPUT = "main"
# The names build() writes itself, so no model file may use them, with what each one holds.
_RESERVED = {OUTPUT: "the compiled program", STDERR_ATTACHMENT: "the compiler's raw stderr"}

# POSIX runs each command in its own session, so a timeout can kill the compiler's child processes too.
_POSIX = os.name == "posix"
# How long, in seconds, a timed-out command's process tree gets to go away before its output is taken as is.
_KILL_GRACE_S = 10.0
# The documented output cap: capped_runner keeps at most this many bytes of a command's stdout, and of its stderr.
OUTPUT_CAP_BYTES = 1 << 20
# How many of the kept bytes are the stream's last bytes, so its final lines survive the cap.
OUTPUT_TAIL_BYTES = 4096
# The most a drain thread reads from a pipe at once.
_CHUNK_BYTES = 1 << 16


@dataclass(frozen=True)
class CommandResult:
    """How one command ended: its exit status, its stdout and stderr as text, and whether each was cut.

    `stdout_truncated` and `stderr_truncated` are True when a capped runner
    (capped_runner, at OUTPUT_CAP_BYTES, or a CappedRunner, at its own cap)
    found the stream longer than its cap, so the text holds only its head
    and its last OUTPUT_TAIL_BYTES bytes; the other runners keep everything
    and leave both False.
    """

    returncode: int
    stdout: str
    stderr: str
    stdout_truncated: bool = False
    stderr_truncated: bool = False


# A command runner: (argv, cwd, timeout in seconds) -> CommandResult.
CommandRunner = Callable[[Sequence[str], Path, float], CommandResult]


def subprocess_runner(argv: Sequence[str], cwd: Path, timeout_s: float) -> CommandResult:
    """Run `argv` in `cwd` without a shell and return its status and all of its output; the compilers' default.

    A toolchain built with runner=None uses it. The sandbox executor or the
    stage runner may inject a sandboxed runner instead. A toolchain only
    compiles: model-generated code never runs through it. stdin is empty,
    and stdout and stderr are drained while the command runs, kept whole,
    and decoded as UTF-8 with errors="replace". When the command, or a
    process holding its pipes, runs past `timeout_s`, it and every process
    it started (a compiler driver starts its stages, such as cicc, ptxas,
    and the host compiler) are killed, and the result has returncode -1 and
    the output read so far, with a last stderr line "timed out after N s".
    The command inherits the parent's environment; EnvRunner gives it a
    chosen one, and capped_runner keeps only a capped part of the output.
    """
    return _run_command(argv, cwd, timeout_s, None, None)


def capped_runner(argv: Sequence[str], cwd: Path, timeout_s: float) -> CommandResult:
    """Run `argv` as subprocess_runner does, but keep at most OUTPUT_CAP_BYTES of stdout and of stderr.

    This is the sandbox's default runner (lassi.executors.sandbox, P0.16 R4):
    whatever generated code prints, the runner keeps at most the cap of
    either stream. Past the cap it keeps the stream's head and its last
    OUTPUT_TAIL_BYTES bytes and sets the stream's truncation flag (see the
    module docstring). Timeouts work as in subprocess_runner, so after one
    the returned stderr is the kept bytes plus the runner's own "timed out"
    line. Decoding the kept bytes makes short-lived copies, so the runner's
    peak memory for the output is a small multiple of the cap.
    """
    return _run_command(argv, cwd, timeout_s, None, OUTPUT_CAP_BYTES)


class CappedRunner:
    """A CommandRunner like capped_runner at a cap of its own: at most `cap_bytes` of stdout and of stderr.

    Past the cap a stream keeps its first cap_bytes - OUTPUT_TAIL_BYTES bytes
    and its last OUTPUT_TAIL_BYTES bytes, and the result's truncation flag
    for that stream is set, as with capped_runner; the command inherits the
    parent's environment. The compile sandbox runs every compile through one
    (lassi.executors.sandbox COMPILE_OUTPUT_CAP_BYTES, P0.20), since the
    runner's buffers lie outside the sandbox's memory limit. The runner's
    peak memory for the output is a small multiple of the cap.
    """

    def __init__(self, cap_bytes: int) -> None:
        """Keep the cap in bytes; ValueError unless it is an int (not a bool) above OUTPUT_TAIL_BYTES."""
        if isinstance(cap_bytes, bool) or not isinstance(cap_bytes, int) or cap_bytes <= OUTPUT_TAIL_BYTES:
            raise ValueError(
                f"cap_bytes must be an integer above OUTPUT_TAIL_BYTES ({OUTPUT_TAIL_BYTES}), got {cap_bytes!r}"
            )
        self.cap_bytes = cap_bytes

    def __repr__(self) -> str:
        """Show the class and the cap."""
        return f"{type(self).__name__}(cap_bytes={self.cap_bytes})"

    def __call__(self, argv: Sequence[str], cwd: Path, timeout_s: float) -> CommandResult:
        """Run `argv` in `cwd` as capped_runner does, keeping at most cap_bytes of each stream."""
        return _run_command(argv, cwd, timeout_s, None, self.cap_bytes)


class EnvRunner:
    """A CommandRunner that behaves like subprocess_runner but gives the command exactly `env` as its environment.

    Nothing from the parent's environment reaches the command unless `env`
    holds it, so variables that change a compile silently (NVCC_PREPEND_FLAGS,
    NVCC_APPEND_FLAGS, CPATH, and the like) stay out. The stage runner runs
    git with one; since P0.20 it compiles, and checks each pinned compiler's
    --version, through lassi.executors.sandbox.SandboxedCompileRunner instead
    (lassi.core.runner). A copy of `env` is kept, so a later change to the
    caller's mapping changes nothing.
    """

    def __init__(self, env: Mapping[str, str]) -> None:
        """Keep a copy of the environment every command gets."""
        self.env: dict[str, str] = dict(env)

    def __repr__(self) -> str:
        """Show the class and the environment's variable names only, never their values."""
        return f"{type(self).__name__}(names={sorted(self.env)!r})"

    def __call__(self, argv: Sequence[str], cwd: Path, timeout_s: float) -> CommandResult:
        """Run `argv` in `cwd` as subprocess_runner does (all output kept), with this environment and nothing else."""
        return _run_command(argv, cwd, timeout_s, self.env, None)


class _CappedPipe:
    """Drains one pipe on a daemon thread and keeps at most `cap` bytes of it, its head and its tail, or all of it."""

    def __init__(self, stream: IO[bytes] | None, cap: int | None) -> None:
        """Start draining `stream`; None (no pipe) reads as empty. `cap` None keeps everything."""
        self._stream = stream
        self._cap = cap
        self._lock = threading.Lock()
        self._head = bytearray()
        self._tail = bytearray()
        self._total = 0
        self._abandoned = False
        self._thread = threading.Thread(target=self._drain, name="lassi-output-drain", daemon=True)
        if stream is not None:
            self._thread.start()

    def _drain(self) -> None:
        """Read the pipe in chunks until it closes, keeping what the cap allows."""
        assert self._stream is not None
        read = getattr(self._stream, "read1", self._stream.read)
        try:
            while True:
                chunk = read(_CHUNK_BYTES)
                if not chunk:
                    return
                self._keep(chunk)
        except (OSError, ValueError):
            return

    def _keep(self, chunk: bytes) -> None:
        """Add `chunk` to the head until it is full, then keep only the last OUTPUT_TAIL_BYTES of the rest.

        Without a cap, every chunk goes to the head. Once the pipe is
        abandoned, chunks are dropped and not counted.
        """
        with self._lock:
            if self._abandoned:
                return
            self._total += len(chunk)
            if self._cap is None:
                self._head += chunk
                return
            room = self._cap - OUTPUT_TAIL_BYTES - len(self._head)
            if room > 0:
                self._head += chunk[:room]
                chunk = chunk[room:]
            if len(chunk) >= OUTPUT_TAIL_BYTES:
                self._tail = bytearray(chunk[-OUTPUT_TAIL_BYTES:])
            elif chunk:
                self._tail += chunk
                del self._tail[: max(0, len(self._tail) - OUTPUT_TAIL_BYTES)]

    def join(self, timeout_s: float) -> bool:
        """Wait up to `timeout_s` for the pipe to close; return True when it has been drained to its end."""
        if self._thread.is_alive():
            self._thread.join(max(0.0, timeout_s))
        return not self._thread.is_alive()

    def abandon(self) -> None:
        """Keep nothing more: a thread still draining the pipe goes on reading, so no writer blocks, but drops it.

        After a timeout, a process that escaped the kill may hold the pipe
        open and keep writing for as long as it lives; without this, an
        uncapped pipe would grow for that long after the result was taken.
        """
        with self._lock:
            self._abandoned = True

    def close(self) -> None:
        """Close the pipe once its thread has finished; a pipe still being read is left to its thread."""
        if self._stream is not None and not self._thread.is_alive():
            self._stream.close()

    def text(self) -> tuple[str, bool]:
        """Return what was kept, decoded as UTF-8 with errors replaced, and whether anything was dropped."""
        with self._lock:
            data = bytes(self._head) + bytes(self._tail)
            truncated = self._cap is not None and self._total > self._cap
        return data.decode("utf-8", errors="replace"), truncated


def _run_command(
    argv: Sequence[str], cwd: Path, timeout_s: float, env: Mapping[str, str] | None, cap: int | None
) -> CommandResult:
    """Run `argv` as subprocess_runner describes; `env` None inherits the parent's environment, `cap` None keeps all."""
    process = subprocess.Popen(
        list(argv),
        cwd=cwd,
        env=None if env is None else dict(env),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=_POSIX,
    )
    pipes = (_CappedPipe(process.stdout, cap), _CappedPipe(process.stderr, cap))
    try:
        finished = _finish(process, pipes, timeout_s)
        if not finished:
            _kill_tree(process)
            _drain_after_kill(process, pipes)
    except BaseException:
        _kill_tree(process)
        raise
    finally:
        for pipe in pipes:
            pipe.abandon()
            pipe.close()
    (stdout, stdout_cut), (stderr, stderr_cut) = (pipe.text() for pipe in pipes)
    if not finished:
        if stderr and not stderr.endswith("\n"):
            stderr += "\n"
        stderr += f"timed out after {timeout_s:g} s\n"
        return CommandResult(-1, stdout, stderr, stdout_truncated=stdout_cut, stderr_truncated=stderr_cut)
    return CommandResult(process.returncode, stdout, stderr, stdout_truncated=stdout_cut, stderr_truncated=stderr_cut)


def _finish(process: subprocess.Popen[bytes], pipes: Sequence[_CappedPipe], timeout_s: float) -> bool:
    """Wait for the command to exit and both pipes to close; return False when `timeout_s` ran out first.

    A process the command started may hold the pipes open after the command
    itself exited; that also counts as running past the timeout.
    """
    deadline = time.monotonic() + timeout_s
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return False
    return all(pipe.join(deadline - time.monotonic()) for pipe in pipes)


def _drain_after_kill(process: subprocess.Popen[bytes], pipes: Sequence[_CappedPipe]) -> None:
    """Wait at most _KILL_GRACE_S in all for a killed command to exit and its pipes to close.

    A process outside the killed tree may still hold the pipes open; the
    output is then what was read so far.
    """
    deadline = time.monotonic() + _KILL_GRACE_S
    try:
        process.wait(timeout=_KILL_GRACE_S)
    except subprocess.TimeoutExpired:
        pass
    for pipe in pipes:
        pipe.join(deadline - time.monotonic())


def _kill_tree(process: subprocess.Popen[bytes]) -> None:
    """Kill `process` and the processes it started: its session on POSIX, its process tree on Windows."""
    if _POSIX:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    else:
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=_KILL_GRACE_S,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
    process.kill()


class CompilerToolchain:
    """A Toolchain that compiles its source files with one command (bible Component Interfaces, Toolchain).

    Subclasses set the class attributes `name`, `capabilities`, and
    `SOURCE_SUFFIXES`, and implement command() and parse(); one may narrow
    the sources with _sources(). The raw stderr is
    kept as the attachment STDERR_ATTACHMENT in the workdir and appears
    nowhere else in the BuildResult.
    """

    name: str
    capabilities: frozenset[str]
    SOURCE_SUFFIXES: tuple[str, ...] = ()

    def __init__(self, *, executable: str, runner: CommandRunner | None, timeout_s: float) -> None:
        """Keep the compiler executable, the runner (None means subprocess_runner), and the timeout in seconds."""
        self.executable = executable
        self.runner: CommandRunner = subprocess_runner if runner is None else runner
        self.timeout_s = timeout_s

    def command(self, sources: Sequence[str]) -> list[str]:
        """Return the compiler command line for `sources`; each subclass defines it."""
        raise NotImplementedError

    def parse(self, stderr: str, files: Mapping[str, str]) -> list[Diagnostic]:
        """Return the Diagnostics in the compiler's `stderr`; each subclass defines it."""
        raise NotImplementedError

    def build(self, files: Mapping[str, str], workdir: Path, harness: Mapping[str, str] | None = None) -> BuildResult:
        """Write `files` (relative path -> text) under `workdir`, compile the sources, and parse the diagnostics.

        `harness` holds the bench item's support files (relative path ->
        text, such as a header its sources include), written beside `files`
        and never compiled as sources. A model file at a harness file's path is
        refused with a "bad-path" error before anything is written, so the
        model can never replace a harness file.

        Every path is checked as lassi.core.files checks it and every text is
        encoded before anything is written, so a bad path raises ValueError
        with the workdir untouched (parse_file_blocks never yields one). A
        file set the workdir cannot hold (a path that another path needs as
        a directory, or a path under the reserved names OUTPUT and
        STDERR_ATTACHMENT) gives "bad-path" errors, and nothing is written or
        run. Files are written as exact UTF-8 bytes. The sources are the
        files _sources() selects, by default those whose suffix is in
        SOURCE_SUFFIXES, in sorted order; with none, the result is one
        "no-sources" error and nothing runs. Otherwise a
        stale workdir/OUTPUT is removed, the raw stderr goes to
        STDERR_ATTACHMENT, and the artifact is workdir/OUTPUT when the status
        is 0 and the compiler wrote it. A failed build always has an error:
        "exit-status" when no error was parsed or the command timed out, and
        "no-artifact" when the status is 0 but no artifact was written. The
        workdir should be fresh for each attempt, since files left by an
        earlier attempt stay visible to the compiler.
        """
        workdir = Path(workdir)
        harness = dict(harness or {})
        for path in [*files, *harness]:
            _check_path(path)
        payloads = {path: text.encode("utf-8") for path, text in [*files.items(), *harness.items()]}
        refused = self._unwritable(files, harness)
        if refused:
            return BuildResult(artifact=None, diagnostics=refused, stderr_ref="")
        for path, data in payloads.items():
            target = workdir / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        sources = self._sources(files)
        if not sources:
            return BuildResult(artifact=None, diagnostics=[self._no_sources()], stderr_ref="")
        output = workdir / OUTPUT
        if output.is_file() or output.is_symlink():
            output.unlink()
        result = self.runner(self.command(sources), workdir, self.timeout_s)
        (workdir / STDERR_ATTACHMENT).write_bytes(result.stderr.encode("utf-8"))
        diagnostics = self.parse(result.stderr, files)
        artifact = output if result.returncode == 0 and output.is_file() else None
        failure = self._failure(result.returncode, artifact, diagnostics)
        if failure is not None:
            diagnostics.append(failure)
        return BuildResult(artifact=artifact, diagnostics=diagnostics, stderr_ref=STDERR_ATTACHMENT)

    def _sources(self, files: Mapping[str, str]) -> list[str]:
        """Return the paths of `files` the compiler builds, in sorted order: those whose suffix is in SOURCE_SUFFIXES.

        A subclass may select fewer (lassi.toolchains.ttmetal_build leaves
        kernel sources out); harness files are never sources.
        """
        return sorted(path for path in files if PurePosixPath(path).suffix in self.SOURCE_SUFFIXES)

    def _unwritable(self, files: Mapping[str, str], harness: Mapping[str, str]) -> list[Diagnostic]:
        """Return a "bad-path" error for each path the workdir cannot hold, in sorted order.

        A path is refused when its first segment is OUTPUT or
        STDERR_ATTACHMENT, which build() writes itself, when a harness file
        has that path, or when another path, a harness path included, needs
        it as a directory (a file "a" next to "a/b.cu").
        """
        every = [*files, *harness]
        directories = {parent.as_posix() for path in every for parent in PurePosixPath(path).parents}
        refused: list[Diagnostic] = []
        for path in sorted(set(every)):
            first = path.split("/")[0]
            if first in _RESERVED:
                reason = f"the build keeps the name {first!r} for {_RESERVED[first]}"
            elif path in files and path in harness:
                reason = "the harness provides that file for the build"
            elif path in directories:
                reason = "other files use it as a directory"
            else:
                continue
            message = f"the file {path!r} cannot be written: {reason}. Rename it and return the files again"
            refused.append(compile_diagnostic("error", message, code="bad-path", file=path))
        return refused

    def _failure(self, returncode: int, artifact: Path | None, diagnostics: list[Diagnostic]) -> Diagnostic | None:
        """Return the error a failed build adds to its parsed `diagnostics`, or None when none is needed.

        A timeout (status -1) is always reported. Otherwise a failed build
        with no parsed error gets "exit-status" for a nonzero status and
        "no-artifact" for status 0 without an artifact.
        """
        if returncode == -1:
            return self._exit_status(returncode)
        if artifact is not None or any(diagnostic.severity == "error" for diagnostic in diagnostics):
            return None
        if returncode != 0:
            return self._exit_status(returncode)
        message = f"{self._tool()} exited with status 0 but did not write the program file {OUTPUT!r}"
        return compile_diagnostic("error", message, code="no-artifact")

    def _no_sources(self) -> Diagnostic:
        """Return the error for a build with no source file."""
        suffixes = ", ".join(self.SOURCE_SUFFIXES)
        message = (
            f"no source file to compile: {self.name} compiles files ending in {suffixes}; return at least one, "
            "in a fenced block whose first line is // FILE: <path>"
        )
        return compile_diagnostic("error", message, code="no-sources")

    def _exit_status(self, returncode: int) -> Diagnostic:
        """Return the error for a timed-out build, or a failed one whose stderr held no error the parser reads."""
        if returncode == -1:
            message = f"{self._tool()} timed out after {self.timeout_s:g} s and was stopped (status -1)"
        else:
            message = f"{self._tool()} exited with status {returncode} and printed no error that could be parsed"
        return compile_diagnostic("error", message, code="exit-status")

    def _tool(self) -> str:
        """Return how messages name the tool: the executable's file name and the preset, as "nvcc (nvcc-sm80)"."""
        return f"{PurePath(self.executable).name} ({self.name})"
