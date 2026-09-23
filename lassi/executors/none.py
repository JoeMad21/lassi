"""The compile-only executor, registered as Executor "none" (bible Execution Backends, executor none).

A compile-only tier builds artifacts but runs nothing, so this executor never
touches the artifact or its inputs and starts no process. It declares only
`compile_only`, so a stage that requires `runs_code` fails at recipe load
(bible Component Interfaces, capability validation).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from lassi.core.interfaces import Limits, RunResult
from lassi.core.registry import register


@register("Executor", "none")
class NoneExecutor:
    """Runs nothing: every result has exit_code None and no output.

    hang=False and wall_s=0.0 are there only because RunResult needs a bool
    and a float; they are not measurements. A compile-only trial keeps the
    Result Record's RunInfo exit_code, hang, and wall_s as None.
    """

    name = "none"
    capabilities = frozenset({"compile_only"})

    def run(self, artifact: Path, inputs: Sequence[str], limits: Limits) -> RunResult:
        """Return the compile-only RunResult; `artifact`, `inputs`, and `limits` are not used."""
        return RunResult(exit_code=None, hang=False, stdout="", stderr="", output_files={}, wall_s=0.0)
