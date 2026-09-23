"""The `lassi` command line: `lassi run <recipe>` runs a recipe with the stage runner (lassi.core.runner).

    lassi run <recipe> [--runs-root P] [--run-id ID] [--bench-root P]

The run tree goes under <runs root>/runs/<run id>; the runs root defaults to
$LASSI_RUNS_ROOT, then the recipe's runs_root, and the bench sources to the
suite's fetched sources under $LASSI_SCRATCH (see lassi.core.runner). The
command prints one line per trial and the run directory. It exits 0 on
success, and 2 with the message on stderr when the recipe does not load
(RecipeError), the run cannot start (RunError), or the sandbox is not
available (SandboxUnavailableError). Importing this module registers every
component, since it imports the runner.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from lassi.core.recipe import RecipeError
from lassi.core.runner import RunError, RunOptions, run_recipe
from lassi.executors import SandboxUnavailableError


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser: the `run` subcommand and its options."""
    parser = argparse.ArgumentParser(prog="lassi", description="Run LASSI recipes.")
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")
    run = commands.add_parser("run", help="run a recipe and write its run tree")
    run.add_argument("recipe", type=Path, help="the recipe file to run")
    run.add_argument("--runs-root", type=Path, help="the runs root (default: $LASSI_RUNS_ROOT, then the recipe's)")
    run.add_argument("--run-id", help="the run directory's name (default: the UTC start time, YYYYMMDD-HHMMSS)")
    run.add_argument("--bench-root", type=Path, help="the suite's fetched sources (default: under $LASSI_SCRATCH)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line with `argv` (default: sys.argv[1:]) and return the exit status."""
    args = build_parser().parse_args(None if argv is None else list(argv))
    options = RunOptions(runs_root=args.runs_root, run_id=args.run_id, bench_root=args.bench_root)
    try:
        run_recipe(args.recipe, options)
    except (RecipeError, RunError, SandboxUnavailableError) as error:
        print(f"lassi {args.command}: {error}", file=sys.stderr)
        return 2
    return 0
