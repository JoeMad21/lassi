"""List, record, and select the recipes of a suite run (plans/runs/suite/run-suite.sh's helper).

Usage, from the repository root of a checkout (run-suite.sh runs it with `uv run python`):

    python plans/runs/suite/recipes.py manifest --out TSV --resolved DIR --sweep DIR... --model SLUG... [--only GLOB...]
    python plans/runs/suite/recipes.py needs --manifest TSV
    python plans/runs/suite/recipes.py select --manifest TSV --sweep DIR --model SLUG [--failed ASSET...]

manifest loads every recipe <sweep>/<slug>/<name>.yaml of each sweep and model with the repository's recipe
loader (lassi.core.recipe.load_recipe) and writes one TSV row per recipe: the sweep, the model slug, the recipe
name (<combo>-<half>), whether it is selected (every recipe when no --only is given, else those whose name
matches one of the globs), its recipe_hash, the model id it names, the generated assets it needs, and its path.
For every selected recipe it also writes the resolved recipe (lassi.core.recipe.resolved_yaml, the text a run
saves) to DIR/<sweep>/<slug>/<name>.yaml, the recipes' provenance beside the run. A recipe that does not load
stops it with exit status 2 and the loader's message.

needs prints each asset the selected recipes need, one per line: context:<pack>, examples:<pack>, and
retrieval:<index>, the assets run-suite.sh builds on the host before the sweep.

select prints one line, "<count>\\t<pattern>": the number of recipes of one sweep and model that run, and the
extended regular expression that leaves out the others (unselected, or needing an asset named by --failed), for
run.sh's SWEEP_SKIP, which it matches against each recipe's name; the pattern is empty when none is left out.
Nothing here runs a recipe, a model, or a device.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import sys
from pathlib import Path
from typing import Any

HEADER = ("sweep", "slug", "recipe", "selected", "recipe_hash", "model_id", "needs", "path")


def needs_of(data: dict[str, Any]) -> list[str]:
    """Return the generated assets a resolved recipe mapping needs: context packs, examples, retrieval index."""
    packs = data.get("context") or []
    found = [f"context:{name}" for name in ([packs] if isinstance(packs, str) else packs)]
    examples = data.get("examples")
    if isinstance(examples, dict) and examples.get("pack"):
        found.append(f"examples:{examples['pack']}")
    retrieval = data.get("retrieval")
    if isinstance(retrieval, dict) and retrieval.get("index"):
        found.append(f"retrieval:{retrieval['index']}")
    return found


def manifest(args: argparse.Namespace) -> int:
    """Write the manifest TSV and the selected recipes' resolved text; return 0, or 2 when a recipe fails to load."""
    from lassi.core import runner  # noqa: F401  (importing the runner registers every component)
    from lassi.core.recipe import RecipeError, load_recipe, resolved_yaml

    rows = []
    for sweep in args.sweep:
        for slug in args.model:
            for path in sorted(Path(sweep, slug).glob("*.yaml")):
                try:
                    recipe = load_recipe(path)
                except RecipeError as exc:
                    print(f"recipes.py: {path.as_posix()} does not load: {exc}", file=sys.stderr)
                    return 2
                name = path.stem
                chosen = not args.only or any(fnmatch.fnmatchcase(name, glob) for glob in args.only)
                rows.append([sweep, slug, name, "yes" if chosen else "no", recipe.recipe_hash,
                             str(recipe.data.get("model", {}).get("id", "-")),
                             ",".join(needs_of(recipe.data)) or "-", path.as_posix()])
                if chosen:
                    target = Path(args.resolved, sweep.strip("/").replace("/", "_"), slug, f"{name}.yaml")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(resolved_yaml(recipe).encode("ascii"))
    with open(args.out, "w", encoding="ascii", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(HEADER)
        writer.writerows(rows)
    chosen = sum(1 for row in rows if row[3] == "yes")
    print(f"recipes.py: {len(rows)} recipes listed, {chosen} selected, into {args.out}", file=sys.stderr)
    return 0


def read_manifest(path: str) -> list[dict[str, str]]:
    """Return the manifest's rows as dicts keyed by HEADER."""
    with open(path, encoding="ascii", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def needs(args: argparse.Namespace) -> int:
    """Print every asset the selected recipes need, sorted, one per line."""
    found = {need for row in read_manifest(args.manifest) if row["selected"] == "yes"
             for need in row["needs"].split(",") if need != "-"}
    for need in sorted(found):
        print(need)
    return 0


def select(args: argparse.Namespace) -> int:
    """Print the count of recipes of one sweep and model that run, a tab, and the SWEEP_SKIP pattern."""
    failed = set(args.failed)
    run, skip = [], []
    for row in read_manifest(args.manifest):
        if row["sweep"] != args.sweep or row["slug"] != args.model:
            continue
        lacking = failed.intersection(row["needs"].split(","))
        (run if row["selected"] == "yes" and not lacking else skip).append(row["recipe"])
    names = "|".join(name.replace(".", "\\.") for name in sorted(skip))
    print(f"{len(run)}\t{'^(' + names + ')$' if skip else ''}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Return the helper's argument parser."""
    parser = argparse.ArgumentParser(prog="recipes.py", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    made = sub.add_parser("manifest", help="load, list, and record the recipes")
    made.add_argument("--out", required=True, help="the manifest TSV to write")
    made.add_argument("--resolved", required=True, help="where the selected recipes' resolved text goes")
    made.add_argument("--sweep", action="append", required=True, help="a recipe directory (repeatable)")
    made.add_argument("--model", action="append", required=True, help="a model slug (repeatable)")
    made.add_argument("--only", action="append", default=[], help="a glob over recipe names (repeatable)")
    listed = sub.add_parser("needs", help="print the assets the selected recipes need")
    listed.add_argument("--manifest", required=True, help="the manifest TSV")
    picked = sub.add_parser("select", help="print the count and SWEEP_SKIP pattern of one sweep and model")
    picked.add_argument("--manifest", required=True, help="the manifest TSV")
    picked.add_argument("--sweep", required=True, help="the recipe directory, as given to manifest")
    picked.add_argument("--model", required=True, help="the model slug")
    picked.add_argument("--failed", action="append", default=[], help="an asset that did not build (repeatable)")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one helper command; return its exit status."""
    args = build_parser().parse_args(argv)
    return {"manifest": manifest, "needs": needs, "select": select}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
