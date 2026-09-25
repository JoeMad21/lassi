"""The `lassi` command line: `lassi run` runs a recipe, and `lassi score` scores a finished run.

    lassi run <recipe> [--runs-root P] [--run-id ID] [--bench-root P]
    lassi score <run dir> --profile NAME [--profile NAME ...] [--score-id ID] [--bench-root P] [--runs-root P]

`lassi run` runs the recipe with the stage runner (lassi.core.runner). The
run tree goes under <runs root>/runs/<run id>; the runs root defaults to
$LASSI_RUNS_ROOT, then the recipe's runs_root, and the bench sources to the
suite's fetched sources under $LASSI_SCRATCH. The command prints one line per
trial and the run directory. It exits 0 on success, and 2 with the message
on stderr when the recipe does not load (RecipeError), the run cannot start
(RunError), or the sandbox is not available (SandboxUnavailableError).

`lassi score` scores the run tree at <run dir> with each registered
ScoreProfile named, in the order given, and writes the review packet under
<runs root>/scores/<score id> (lassi.scoring.score_run); it never changes
the run tree it reads. The runs root defaults to $LASSI_RUNS_ROOT, then the
run dir's grandparent when its parent is named `runs`; the bench sources, for
a profile that reads them, to the suite's fetched sources under
$LASSI_SCRATCH. It prints the score directory and exits 0, or exits 2 with
`lassi score: <message>` on stderr when the pass is refused (ScoreError).
Profile names are checked against the registry by the pass itself, never by
argparse, so an unknown one is a refusal like any other.

Importing this module registers every component, since it imports the runner
and the scoring package.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from lassi.core.recipe import RecipeError
from lassi.core.runner import RunError, RunOptions, run_recipe
from lassi.executors import SandboxUnavailableError
from lassi.scoring.score_run import ScoreError, score_run


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser: the `run` and `score` subcommands and their options."""
    parser = argparse.ArgumentParser(prog="lassi", description="Run LASSI recipes and score finished runs.")
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")
    run = commands.add_parser("run", help="run a recipe and write its run tree")
    run.add_argument("recipe", type=Path, help="the recipe file to run")
    run.add_argument("--runs-root", type=Path, help="the runs root (default: $LASSI_RUNS_ROOT, then the recipe's)")
    run.add_argument("--run-id", help="the run directory's name (default: the UTC start time, YYYYMMDD-HHMMSS)")
    run.add_argument("--bench-root", type=Path, help="the suite's fetched sources (default: under $LASSI_SCRATCH)")
    score = commands.add_parser("score", help="score a finished run and write its review packet")
    score.add_argument("run_dir", type=Path, help="the run directory, <runs root>/runs/<run id>")
    score.add_argument(
        "--profile", dest="profiles", action="append", metavar="NAME",
        help="a registered ScoreProfile to score with; repeat the option for more, in order",
    )
    score.add_argument("--score-id", help="the score directory's name (default: the UTC time, YYYYMMDD-HHMMSS)")
    score.add_argument(
        "--bench-root", type=Path,
        help="the suite's fetched sources, for a profile that reads them (default: under $LASSI_SCRATCH)",
    )
    score.add_argument(
        "--runs-root", type=Path,
        help="where scores/<score id> goes (default: $LASSI_RUNS_ROOT, then the run dir's grandparent)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line with `argv` (default: sys.argv[1:]) and return the exit status."""
    args = build_parser().parse_args(None if argv is None else list(argv))
    if args.command == "score":
        return _score(args)
    return _run(args)


def _run(args: argparse.Namespace) -> int:
    """Run `lassi run` and return its exit status: 0, or 2 with the message on stderr."""
    options = RunOptions(runs_root=args.runs_root, run_id=args.run_id, bench_root=args.bench_root)
    try:
        run_recipe(args.recipe, options)
    except (RecipeError, RunError, SandboxUnavailableError) as error:
        print(f"lassi {args.command}: {error}", file=sys.stderr)
        return 2
    return 0


def _score(args: argparse.Namespace) -> int:
    """Run `lassi score`, print the score directory, and return 0; or 2 with `lassi score: <message>` on stderr."""
    try:
        out = score_run(
            args.run_dir, args.profiles or [], score_id=args.score_id, bench_root=args.bench_root,
            runs_root=args.runs_root,
        )
    except ScoreError as error:
        print(f"lassi {args.command}: {error}", file=sys.stderr)
        return 2
    print(out)
    return 0
