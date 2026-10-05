"""The native executor, registered as Executor "native" (bible Execution Backends, executor native).

It runs a CPU artifact (a compiled program or a script) with its inputs as
arguments, and only through lassi.executors.sandbox (Agent Rule 6). The
artifact's directory is the run's workdir, the one writable host directory;
it must lie inside the runs root, $LASSI_RUNS_ROOT (lassi.executors.workdir),
so the writable mount is never an ad hoc host directory (Agent Rule 7). The
hidden roots are $LASSI_SCRATCH and $HOME unless configured otherwise, and
always the runs root too, so no other trial is readable wherever the runs
root lies: inside the sandbox they show only the path skeleton to the
workdir, the harness (when configured), and the pinned toolchains root
($LASSI_TOOLCHAINS unless configured otherwise), the last two read-only
(P0.16 R3). Every one of those paths is resolved first, so a symbolic link
cannot carry a hidden root into view under another name. The regular files
the run creates in the workdir are its output files (bible Component
Interfaces, Executor contract rules); symbolic links are left out, so a run
cannot point its caller at a host file outside the workdir. The RunResult
carries the sandbox's stdout and stderr truncation flags (P0.16 R4) and its
workdir_incomplete flag (P0.16 R5).

The program's environment bounds its OpenMP threads by Limits.cpus
(OMP_NUM_THREADS; DEMO.2, plans/spikes/demo-multicore-proxy.md): without the
bound the OpenMP runtime starts one thread per host CPU, which on a host with
as many CPUs as the sandbox's TasksMax fails a thread creation and aborts
the program. See NativeExecutor.run for the exact environment.

NativeExecutor.device() names the device its programs run on, the host CPU,
with the CPU model read from CPUINFO (task P4.5; Agent Rule 1); the runner
records it in provenance.json and every trial's provenance.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from types import MappingProxyType

from lassi.core.interfaces import Limits, RunResult
from lassi.core.registry import register
from lassi.executors.sandbox import SANDBOX_PATH, Sandbox, SandboxSpec, SandboxUnavailableError
from lassi.executors.workdir import runs_root

# The file the host CPU model is read from, and the device name of a native run without a model.
CPUINFO = Path("/proc/cpuinfo")
HOST_CPU = "host CPU (native)"
# The field of CPUINFO that names the CPU model.
_MODEL_FIELD = "model name"

# The environment variables that name the default hidden roots, in order.
_ROOT_VARIABLES = ("LASSI_SCRATCH", "HOME")
# The environment variable that names the default toolchains root, which the sandbox re-exposes read-only.
_TOOLCHAINS_VARIABLE = "LASSI_TOOLCHAINS"
# The fixed part of every program's environment: the sandbox's default program environment without HOME, which
# SandboxSpec.environment never holds (P0.20). _program_environment adds the OpenMP thread bound.
_PROGRAM_ENVIRONMENT = MappingProxyType({"PATH": SANDBOX_PATH, "LANG": "C.UTF-8", "TMPDIR": "/tmp"})


def _program_environment(limits: Limits) -> dict[str, str]:
    """Return the program's environment: _PROGRAM_ENVIRONMENT plus OMP_NUM_THREADS=<limits.cpus>.

    Nothing comes from the caller's environment, a caller's OMP_NUM_THREADS
    included. The sandbox checks limits.cpus (an integer >= 1) before
    anything runs.
    """
    return {**_PROGRAM_ENVIRONMENT, "OMP_NUM_THREADS": str(limits.cpus)}


def _cpu_model() -> str:
    """Return the value of the first "model name" line of CPUINFO as one line of printable ASCII, or "" for none.

    The whitespace is collapsed: surrounding whitespace is dropped and each
    inner run of whitespace becomes one space. Any other character that is
    not printable ASCII becomes its backslash escape (as ascii() writes it),
    so the result has no leading or trailing blank. A file that cannot be
    read, holds no such line, or whose first such line has no value gives "".
    Only the file is read; no process starts.
    """
    try:
        with CPUINFO.open("rb") as handle:
            for raw in handle:
                key, colon, value = raw.decode("utf-8", errors="replace").partition(":")
                if colon and key.strip() == _MODEL_FIELD:
                    return "".join(_printable(character) for character in " ".join(value.split()))
    except OSError:
        return ""
    return ""


def _printable(character: str) -> str:
    """Return `character` when it is printable ASCII, else its backslash escape as ascii() writes it."""
    return character if character.isascii() and character.isprintable() else ascii(character)[1:-1]


def _absolute_variable(name: str, use: str) -> Path | None:
    """Return $name as a Path, or None when it is unset or empty.

    A set value that is not an absolute path raises SandboxUnavailableError
    rather than being skipped, so a bad value never changes what the sandbox
    shows without notice. `use` says what the path is for, in the message.
    """
    value = os.environ.get(name, "")
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        raise SandboxUnavailableError(f"${name} must be an absolute path to be {use}, got {value!r}")
    return path


def _environment_roots() -> tuple[Path, ...]:
    """Return $LASSI_SCRATCH and $HOME, the ones that are set, in that order, without duplicates."""
    roots: list[Path] = []
    for name in _ROOT_VARIABLES:
        path = _absolute_variable(name, "hidden in the sandbox")
        if path is not None and path not in roots:
            roots.append(path)
    return tuple(roots)


def _resolved(path: Path) -> Path:
    """Return `path` with its symbolic links resolved as far as it exists (os.path.realpath)."""
    return Path(os.path.realpath(path))


def _checked_workdir(artifact: Path) -> Path:
    """Return the artifact's directory, resolved, after checking that it lies inside the runs root.

    Resolving first means a symbolic link cannot point the run's one
    writable mount outside the runs root. The runs root itself is refused,
    since every trial's files live there. Raise ValueError otherwise, and
    when $LASSI_RUNS_ROOT is unset or relative (workdir.runs_root).
    """
    root = runs_root().resolve()
    workdir = artifact.parent.resolve()
    if root not in workdir.parents:
        raise ValueError(f"the artifact's directory {str(workdir)!r} must lie inside the runs root {str(root)!r}")
    return workdir


def _regular_files(root: Path) -> dict[str, Path]:
    """Return the regular files under `root`, as relative POSIX path -> path.

    Directories and symbolic links are left out, and linked directories are
    not entered, so a run cannot point its caller at a host file outside the
    workdir.
    """
    found: dict[str, Path] = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            path = Path(dirpath) / name
            if not path.is_symlink() and path.is_file():
                found[path.relative_to(root).as_posix()] = path
    return found


@register("Executor", "native")
class NativeExecutor:
    """Runs a CPU artifact with its inputs as arguments, only inside the sandbox.

    The config key `harness` names a directory mounted read-only at its own
    path. The keyword settings are for the runner and tests: `hidden_roots`
    (None means $LASSI_SCRATCH and $HOME from the environment at run time;
    the runs root is always added), `toolchains` (None means
    $LASSI_TOOLCHAINS at run time; unset or empty means none), where a
    relative environment value raises SandboxUnavailableError, and `sandbox`
    (None means a Sandbox with the default runner).
    """

    name = "native"
    capabilities = frozenset({"runs_code", "sandboxed"})
    config_keys = frozenset({"harness"})

    def __init__(
        self,
        *,
        harness: str | None = None,
        hidden_roots: Sequence[str] | None = None,
        toolchains: str | None = None,
        sandbox: Sandbox | None = None,
    ) -> None:
        """Keep the harness, hidden roots, toolchains root, and sandbox; factory() with no arguments works."""
        if isinstance(hidden_roots, (str, os.PathLike)):
            raise ValueError(f"hidden_roots must be a sequence of paths, got {hidden_roots!r}")
        self.harness: Path | None = None if harness is None else Path(harness)
        self.hidden_roots: tuple[Path, ...] | None = (
            None if hidden_roots is None else tuple(Path(root) for root in hidden_roots)
        )
        self.toolchains: Path | None = None if toolchains is None else Path(toolchains)
        self.sandbox: Sandbox = Sandbox() if sandbox is None else sandbox

    def device(self) -> str:
        """Return the device the programs run on: "host CPU (native): <model>", or HOST_CPU without a model.

        The model is the first "model name" value of CPUINFO with its
        whitespace collapsed (_cpu_model), so the result is one non-empty
        line of printable ASCII with no leading or trailing blank. It reads
        that file only: it starts no process and never uses the sandbox.
        """
        model = _cpu_model()
        return f"{HOST_CPU}: {model}" if model else HOST_CPU

    def run(self, artifact: Path, inputs: Sequence[str], limits: Limits) -> RunResult:
        """Run `artifact` with `inputs` in the sandbox under `limits` and return its RunResult.

        The workdir is the artifact's directory, resolved, and the argv is
        the artifact then the inputs. output_files holds the regular files
        that exist in the workdir after the run and did not before it
        (relative POSIX path -> absolute path); rewritten files, symbolic
        links, and new directories are not listed. The hidden roots are the
        configured or environment roots plus the runs root, and they, the
        harness, and the toolchains root are resolved (_resolved). A relative
        artifact, a workdir outside the runs root, or a string of inputs
        raises ValueError, and no hidden root from the configuration or the
        environment, or a relative $LASSI_TOOLCHAINS, raises
        SandboxUnavailableError, before anything runs. The spec keeps the
        sandbox's default disk cap.

        The program's environment (SandboxSpec.environment) is exactly
        PATH=SANDBOX_PATH, LANG=C.UTF-8, TMPDIR=/tmp, and
        OMP_NUM_THREADS=<limits.cpus>, which bounds the program's OpenMP
        threads by the run's CPU count (DEMO.2): it is the OpenMP runtime's
        default thread count, so a program's own num_threads clause or
        omp_set_num_threads call can still ask for more, up to the sandbox's
        TasksMax. OMP_NUM_THREADS is a thread count, not a secret or a
        loader variable (Agent Rule 12). The one change from the sandbox's
        default program environment: there is no HOME (the default sets
        HOME=<workdir>), since SandboxSpec.environment never holds HOME; the
        working directory is still the workdir. The
        program runs as `env -i -- NAME=value... <artifact> <inputs>` inside
        the sandbox, so an artifact path that holds "=" raises ValueError,
        and env's own statuses (125, 126, 127) can come back as the exit
        code, as the sandbox module docstring says.
        """
        artifact = Path(artifact)
        if not artifact.is_absolute():
            raise ValueError(f"the artifact must be an absolute path, got {str(artifact)!r}")
        if isinstance(inputs, str):
            raise ValueError(f"inputs must be a sequence of strings, got {inputs!r}")
        workdir = _checked_workdir(artifact)
        roots = _environment_roots() if self.hidden_roots is None else self.hidden_roots
        if not roots:
            raise SandboxUnavailableError(
                "no hidden root: set $LASSI_SCRATCH or $HOME, or pass hidden_roots; nothing ran"
            )
        toolchains = self.toolchains
        if toolchains is None:
            toolchains = _absolute_variable(_TOOLCHAINS_VARIABLE, "mounted read-only in the sandbox")
        spec = SandboxSpec(
            workdir=workdir,
            hidden_roots=(*(_resolved(root) for root in roots), runs_root().resolve()),
            harness=None if self.harness is None else _resolved(self.harness),
            toolchains=None if toolchains is None else _resolved(toolchains),
            environment=_program_environment(limits),
        )
        before = _regular_files(spec.workdir)
        result = self.sandbox.run(spec, [str(artifact), *inputs], limits)
        after = _regular_files(spec.workdir)
        outputs = {relative: after[relative] for relative in sorted(after) if relative not in before}
        return RunResult(
            exit_code=result.returncode,
            hang=result.hang,
            stdout=result.stdout,
            stderr=result.stderr,
            output_files=outputs,
            wall_s=result.wall_s,
            stdout_truncated=result.stdout_truncated,
            stderr_truncated=result.stderr_truncated,
            workdir_incomplete=result.workdir_incomplete,
        )
