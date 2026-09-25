"""The `lassi` command line: `lassi run` runs a recipe, and `lassi score` scores a finished run.

    lassi [--graphics on|off] run <recipe> [--runs-root P] [--run-id ID] [--bench-root P]
    lassi [--graphics on|off] score <run dir> --profile NAME [--profile NAME ...] [--score-id ID]
          [--bench-root P] [--runs-root P]
    lassi [--graphics on|off] settings graphics [on|off]

Every command first decides terminal graphics (lassi.present.settings,
choose_graphics: the --graphics option, LASSI_GRAPHICS, off without an
interactive terminal, the saved preset, the first-run prompt, off). A
--graphics value other than on or off is refused by the argument parser
(argparse choices), which prints its usage message and exits 2. A bad
LASSI_GRAPHICS value or a preset error (a preset that load_preset refuses,
or one whose place cannot be checked) exits 2 with
`lassi <command>: <message>` on stderr before the command starts, except
that `lassi settings graphics on|off` takes a preset error as graphics off
and replaces the preset. With graphics on the command first prints the
banner (lassi.present.banner) to stderr; with graphics off it prints exactly
what it printed without them.

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

`lassi settings graphics on|off` saves the graphics preset and prints where;
`lassi settings graphics` prints the setting that applies on a terminal and
its source, never asking. Both exit 0, or 2 with `lassi settings: <message>`
on stderr for a bad LASSI_GRAPHICS value or a preset error (a preset refused
on the root filesystem, one whose place cannot be checked, one that cannot
be saved, or, when showing, one that load_preset refuses). A value other
than on or off, as in `lassi settings graphics maybe`, is refused by the
argument parser (argparse choices) with its usage message and exit 2, like
a bad --graphics value.

Importing this module registers every component, since it imports the runner
and the scoring package.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from lassi.core.recipe import RecipeError
from lassi.core.runner import RunError, RunOptions, run_recipe
from lassi.executors import SandboxUnavailableError
from lassi.present.banner import print_banner
from lassi.present.settings import (
    ROOT_REFUSAL,
    SettingsError,
    choose_graphics,
    preset_path,
    resolve_graphics,
    save_preset,
)
from lassi.scoring.score_run import ScoreError, score_run


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser: the global --graphics option and the `run`, `score`, and `settings` commands."""
    parser = argparse.ArgumentParser(prog="lassi", description="Run LASSI recipes and score finished runs.")
    parser.add_argument(
        "--graphics", choices=("on", "off"),
        help="terminal graphics for this command (default: $LASSI_GRAPHICS, then the saved preset; off without a "
        "terminal)",
    )
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
    settings = commands.add_parser("settings", help="show or change the user's lassi settings")
    names = settings.add_subparsers(dest="setting", required=True, metavar="setting")
    graphics = names.add_parser("graphics", help="show the graphics setting, or save the preset with on or off")
    graphics.add_argument("value", nargs="?", choices=("on", "off"), help="the preset to save (default: show)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line with `argv` (default: sys.argv[1:]) and return the exit status."""
    args = build_parser().parse_args(None if argv is None else list(argv))
    try:
        on = _decide_graphics(args)
    except SettingsError as error:
        print(f"lassi {args.command}: {error}", file=sys.stderr)
        return 2
    if on:
        print_banner(sys.stderr)
    if args.command == "settings":
        return _settings(args)
    if args.command == "score":
        return _score(args)
    return _run(args)


def _decide_graphics(args: argparse.Namespace) -> bool:
    """Decide graphics for this command with choose_graphics; the settings command never asks.

    `lassi settings graphics on|off` replaces the preset, so there a preset
    that cannot be read leaves graphics off rather than stopping its own
    repair; a bad LASSI_GRAPHICS value is still a SettingsError (a bad
    --graphics value never gets here: the argument parser refuses it).
    """
    stdin_isatty, stdout_isatty = _isatty(sys.stdin), _isatty(sys.stdout)
    repair = args.command == "settings" and args.value is not None
    if repair:
        resolve_graphics(
            option=args.graphics, env=os.environ, stdin_isatty=stdin_isatty, stdout_isatty=stdout_isatty, preset=None
        )
    try:
        on, _ = choose_graphics(
            option=args.graphics, env=os.environ, stdin=sys.stdin, out=sys.stdout, stdin_isatty=stdin_isatty,
            stdout_isatty=stdout_isatty, home=Path.home(), prompt=args.command != "settings",
        )
    except SettingsError:
        if not repair:
            raise
        return False
    return on


def _isatty(stream: object) -> bool:
    """Return True when `stream` is an interactive terminal; False for None, a closed stream, or a non-stream."""
    try:
        return bool(stream.isatty())  # type: ignore[attr-defined]
    except (AttributeError, ValueError, OSError):
        return False


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


def _settings(args: argparse.Namespace) -> int:
    """Run `lassi settings graphics [on|off]` and return 0, or 2 with `lassi settings: <message>` on stderr."""
    try:
        lines = _show_graphics(args.graphics) if args.value is None else _save_graphics(args.value == "on")
    except SettingsError as error:
        print(f"lassi {args.command}: {error}", file=sys.stderr)
        return 2
    print("\n".join(lines))
    return 0


def _save_graphics(on: bool) -> list[str]:
    """Save the graphics preset and return the line that says where; SettingsError when it cannot be saved."""
    path = preset_path(os.environ, home=Path.home())
    if path is None:
        raise SettingsError(ROOT_REFUSAL)
    save_preset(path, on)
    return [f"graphics preset saved: {path} (graphics: {'on' if on else 'off'})"]


def _show_graphics(option: str | None) -> list[str]:
    """Return the setting that applies on a terminal, `graphics: on|off`, then its source; nothing is asked."""
    home = Path.home()
    on, source = choose_graphics(
        option=option, env=os.environ, stdin=sys.stdin, out=sys.stdout, stdin_isatty=True, stdout_isatty=True,
        home=home, prompt=False,
    )
    path = preset_path(os.environ, home=home) if source in ("preset", "default") else None
    if source == "preset":
        described = f"the preset {path}"
    elif source == "default":
        described = ROOT_REFUSAL if path is None else f"default; no preset at {path}, so a lassi command asks"
    else:
        described = f"the {'option' if source == '--graphics' else 'variable'} {source}"
    return [f"graphics: {'on' if on else 'off'}", f"source: {described}"]
