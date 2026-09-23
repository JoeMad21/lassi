"""Executors: the Executor components (bible Component Interfaces, Executor row; Execution Backends).

Importing this package registers the executors in
lassi.core.registry.DEFAULT_REGISTRY. So far it holds the compile-only
executor "none"; the sandbox module and the native executor follow in the rest
of task P0.10, and every executor that runs generated code will run it only
through the sandbox (Agent Rule 6).
"""

from lassi.executors import none
from lassi.executors.none import NoneExecutor

__all__ = ["NoneExecutor", "none"]
