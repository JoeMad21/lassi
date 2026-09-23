"""The command runner and the build steps shared by the compiler toolchains.

lassi.toolchains re-exports CommandResult, CommandRunner, subprocess_runner,
and STDERR_ATTACHMENT from here. The adapters in lassi.toolchains.nvcc and
lassi.toolchains.nvcpp subclass CompilerToolchain, which holds build().
"""

from __future__ import annotations

import os
import signal
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePath, PurePosixPath

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


@dataclass(frozen=True)
class CommandResult:
    """How one command ended: its exit status and its stdout and stderr as text."""

    returncode: int
    stdout: str
    stderr: str


# A command runner: (argv, cwd, timeout in seconds) -> CommandResult.
CommandRunner = Callable[[Sequence[str], Path, float], CommandResult]


def subprocess_runner(argv: Sequence[str], cwd: Path, timeout_s: float) -> CommandResult:
    """Run `argv` in `cwd` without a shell and return its status and output; this is the default runner.

    A toolchain built with runner=None uses it. The sandbox executor or the
    stage runner may inject a sandboxed runner instead. A toolchain only
    compiles: model-generated code never runs through it. stdin is empty and
    stdout and stderr are captured and decoded as UTF-8 with errors="replace".
    When the command runs past `timeout_s`, it and every process it started
    (a compiler driver starts its stages, such as cicc, ptxas, and the host
    compiler) are killed, and the result has returncode -1 and the stderr
    captured so far, followed by a last line "timed out after N s".
    """
    process = subprocess.Popen(
        list(argv),
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=_POSIX,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        _kill_tree(process)
        stdout, stderr = _output_after_kill(process)
        partial = _decode(stderr)
        if partial and not partial.endswith("\n"):
            partial += "\n"
        text = f"{partial}timed out after {timeout_s:g} s\n"
        return CommandResult(returncode=-1, stdout=_decode(stdout), stderr=text)
    except BaseException:
        _kill_tree(process)
        raise
    return CommandResult(returncode=process.returncode, stdout=_decode(stdout), stderr=_decode(stderr))


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


def _output_after_kill(process: subprocess.Popen[bytes]) -> tuple[bytes | None, bytes | None]:
    """Return the stdout and stderr read from a killed `process`, waiting at most _KILL_GRACE_S for the pipes.

    A process outside the killed tree may still hold the pipes open; the
    output is then what was read so far (nothing, on Windows).
    """
    try:
        return process.communicate(timeout=_KILL_GRACE_S)
    except subprocess.TimeoutExpired as expired:
        return expired.stdout, expired.stderr


def _decode(data: bytes | None) -> str:
    """Return captured output as text: UTF-8 with errors replaced, and "" when nothing was captured."""
    return "" if data is None else data.decode("utf-8", errors="replace")


class CompilerToolchain:
    """A Toolchain that compiles its source files with one command (bible Component Interfaces, Toolchain).

    Subclasses set the class attributes `name`, `capabilities`, and
    `SOURCE_SUFFIXES`, and implement command() and parse(). The raw stderr is
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

    def build(self, files: Mapping[str, str], workdir: Path) -> BuildResult:
        """Write `files` (relative path -> text) under `workdir`, compile the sources, and parse the diagnostics.

        Every path is checked as lassi.core.files checks it and every text is
        encoded before anything is written, so a bad path raises ValueError
        with the workdir untouched (parse_file_blocks never yields one). A
        file set the workdir cannot hold (a path that another path needs as
        a directory, or a path under the reserved names OUTPUT and
        STDERR_ATTACHMENT) gives "bad-path" errors, and nothing is written or
        run. Files are written as exact UTF-8 bytes. The sources are the
        files whose suffix is in SOURCE_SUFFIXES, in sorted order; with none,
        the result is one "no-sources" error and nothing runs. Otherwise a
        stale workdir/OUTPUT is removed, the raw stderr goes to
        STDERR_ATTACHMENT, and the artifact is workdir/OUTPUT when the status
        is 0 and the compiler wrote it. A failed build always has an error:
        "exit-status" when no error was parsed or the command timed out, and
        "no-artifact" when the status is 0 but no artifact was written. The
        workdir should be fresh for each attempt, since files left by an
        earlier attempt stay visible to the compiler.
        """
        workdir = Path(workdir)
        for path in files:
            _check_path(path)
        payloads = {path: text.encode("utf-8") for path, text in files.items()}
        refused = self._unwritable(files)
        if refused:
            return BuildResult(artifact=None, diagnostics=refused, stderr_ref="")
        for path, data in payloads.items():
            target = workdir / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        sources = sorted(path for path in files if PurePosixPath(path).suffix in self.SOURCE_SUFFIXES)
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

    def _unwritable(self, files: Mapping[str, str]) -> list[Diagnostic]:
        """Return a "bad-path" error for each path the workdir cannot hold, in sorted order.

        A path is refused when its first segment is OUTPUT or
        STDERR_ATTACHMENT, which build() writes itself, or when another path
        needs it as a directory (a file "a" next to "a/b.cu").
        """
        directories = {parent.as_posix() for path in files for parent in PurePosixPath(path).parents}
        refused: list[Diagnostic] = []
        for path in sorted(files):
            first = path.split("/")[0]
            if first in _RESERVED:
                reason = f"the build keeps the name {first!r} for {_RESERVED[first]}"
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
