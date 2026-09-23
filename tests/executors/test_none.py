"""Tests for the compile-only executor, registered as Executor "none" (P0.10, first slice).

The none executor never runs anything (bible Execution Backends: executor none,
compile only). It returns a RunResult that says so: no exit status, no output,
no hang, zero wall time. The sandbox module and the native executor are the
rest of P0.10. No value here is a measurement.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from lassi.core import interfaces, registry


@pytest.fixture
def none_module():
    """Return the lassi.executors.none module (imported here so collection shows a missing module clearly)."""
    from lassi.executors import none

    return none


def test_registered_as_executor_none_with_compile_only(none_module) -> None:
    entry = registry.DEFAULT_REGISTRY.get("Executor", "none")
    assert entry.factory is none_module.NoneExecutor
    assert entry.capabilities == frozenset({"compile_only"})
    assert "runs_code" not in entry.capabilities
    assert entry.config_keys == frozenset()


def test_run_returns_a_compile_only_result_without_touching_the_artifact(none_module, tmp_path: Path) -> None:
    missing = tmp_path / "never-built"
    limits = interfaces.Limits(wall_s=1.0, memory_mb=64, cpus=1)
    result = none_module.NoneExecutor().run(missing, ["--size", "8"], limits)
    assert result == interfaces.RunResult(
        exit_code=None, hang=False, stdout="", stderr="", output_files={}, wall_s=0.0
    )
    assert not missing.exists()


def test_constructs_from_the_registry_with_no_arguments(none_module) -> None:
    executor = registry.DEFAULT_REGISTRY.get("Executor", "none").factory()
    assert executor.name == "none"
    assert list(inspect.signature(none_module.NoneExecutor.run).parameters) == ["self", "artifact", "inputs", "limits"]


def test_module_runs_no_process(none_module) -> None:
    source = inspect.getsource(none_module)
    for forbidden in ("subprocess", "os.system", "Popen", "exec("):
        assert forbidden not in source
