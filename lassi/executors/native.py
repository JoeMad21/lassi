"""The native executor, registered as Executor "native" (bible Execution Backends, executor native).

It runs a CPU artifact (a compiled program or a script) with its inputs as
arguments, and only through lassi.executors.sandbox (Agent Rule 6). The
artifact's directory is the run's workdir, the one writable host directory;
it must lie inside the runs root, $LASSI_RUNS_ROOT (lassi.executors.workdir),
so the writable mount is never an ad hoc host directory (Agent Rule 7). The
read-only roots are $LASSI_SCRATCH and $HOME unless configured otherwise; the
harness, when configured, is mounted read-only. The regular files the run
creates in the workdir are its output files (bible Component Interfaces,
Executor contract rules); symbolic links are left out, so a run cannot point
its caller at a host file outside the workdir.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

from lassi.core.interfaces import Limits, RunResult
from lassi.core.registry import register
from lassi.executors.sandbox import Sandbox, SandboxSpec, SandboxUnavailableError
from lassi.executors.workdir import runs_root

# The environment variables that name the default read-only roots, in order.
_ROOT_VARIABLES = ("LASSI_SCRATCH", "HOME")


def _environment_roots() -> tuple[Path, ...]:
    """Return $LASSI_SCRATCH and $HOME, the ones that are set, in that order, without duplicates.

    A set value that is not an absolute path raises SandboxUnavailableError
    rather than being skipped, so a bad value never leaves its directory
    writable without notice.
    """
    roots: list[Path] = []
    for name in _ROOT_VARIABLES:
        value = os.environ.get(name, "")
        if not value:
            continue
        path = Path(value)
        if not path.is_absolute():
            raise SandboxUnavailableError(f"${name} must be an absolute path to be mounted read-only, got {value!r}")
        if path not in roots:
            roots.append(path)
    return tuple(roots)


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
    path. The keyword settings `readonly_roots` (None means $LASSI_SCRATCH
    and $HOME from the environment at run time, where a relative value
    raises SandboxUnavailableError) and `sandbox` (None means a Sandbox with
    the default runner) are for the runner and tests.
    """

    name = "native"
    capabilities = frozenset({"runs_code", "sandboxed"})
    config_keys = frozenset({"harness"})

    def __init__(
        self,
        *,
        harness: str | None = None,
        readonly_roots: Sequence[str] | None = None,
        sandbox: Sandbox | None = None,
    ) -> None:
        """Keep the harness, the read-only roots, and the sandbox; factory() with no arguments works."""
        if isinstance(readonly_roots, (str, os.PathLike)):
            raise ValueError(f"readonly_roots must be a sequence of paths, got {readonly_roots!r}")
        self.harness: Path | None = None if harness is None else Path(harness)
        self.readonly_roots: tuple[Path, ...] | None = (
            None if readonly_roots is None else tuple(Path(root) for root in readonly_roots)
        )
        self.sandbox: Sandbox = Sandbox() if sandbox is None else sandbox

    def run(self, artifact: Path, inputs: Sequence[str], limits: Limits) -> RunResult:
        """Run `artifact` with `inputs` in the sandbox under `limits` and return its RunResult.

        The workdir is the artifact's directory, resolved, and the argv is
        the artifact then the inputs. output_files holds the regular files
        that exist in the workdir after the run and did not before it
        (relative POSIX path -> absolute path); rewritten files, symbolic
        links, and new directories are not listed. A relative artifact, a
        workdir outside the runs root, or a string of inputs raises
        ValueError, and no read-only root raises SandboxUnavailableError,
        before anything runs.
        """
        artifact = Path(artifact)
        if not artifact.is_absolute():
            raise ValueError(f"the artifact must be an absolute path, got {str(artifact)!r}")
        if isinstance(inputs, str):
            raise ValueError(f"inputs must be a sequence of strings, got {inputs!r}")
        workdir = _checked_workdir(artifact)
        roots = _environment_roots() if self.readonly_roots is None else self.readonly_roots
        if not roots:
            raise SandboxUnavailableError(
                "no read-only root: set $LASSI_SCRATCH or $HOME, or pass readonly_roots; nothing ran"
            )
        spec = SandboxSpec(workdir=workdir, readonly_roots=roots, harness=self.harness)
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
        )
