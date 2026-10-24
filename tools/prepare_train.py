"""Download a train recipe's base into $HF_HOME/hub: the separate, explicit prepare step (2026-10-23).

Usage:

    uv run --extra cpu tools/prepare_train.py <train.yaml> [--dry-run]

lassi train never downloads: a Hub base (<owner>/<name>@<revision>) and a
scratch base's tokenizer load from $HF_HOME/hub only, and lassi train
refuses a snapshot the cache lacks (lassi.train.bases). This command fills
the cache for one recipe with huggingface_hub's snapshot_download, at the
pinned revision, then checks the snapshot as lassi train will
(lassi.train.prepare). On alpha01 it runs only in the owner's window, as its
own command (the owner's direction of 2026-10-23; Agent Rule 17), with
HF_HOME under the scratch root (Agent Rule 7). --dry-run prints what would
be fetched and fetches nothing; a fixture base needs nothing. Exit status:
0 on success, 2 with a message on a refusal.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from lassi.core.recipe import RecipeError, load_train_recipe  # noqa: E402
from lassi.train import bases, prepare  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Prepare the recipe's base; return 0, or 2 with `prepare_train: <message>` on stderr."""
    parser = argparse.ArgumentParser(prog="prepare_train", description=__doc__.splitlines()[0])
    parser.add_argument("recipe", type=Path, help="the train recipe whose base to download")
    parser.add_argument("--dry-run", action="store_true", help="print what would be fetched; fetch nothing")
    args = parser.parse_args(argv)
    try:
        base = bases.parse_base(load_train_recipe(args.recipe).data["base_model"])
        wanted = prepare.plan(base)
        if wanted is None:
            print(f"prepare_train: {base.name} is a fixture; nothing to download")
            return 0
        repo, revision, patterns = wanted
        target = bases.snapshot_dir(os.environ.get("HF_HOME", "") or "$HF_HOME", repo, revision)
        if args.dry_run:
            print(f"prepare_train: dry run: would fetch {repo} at {revision} ({', '.join(patterns)}) into {target}")
            return 0
        found = prepare.prepare(args.recipe)
    except (OSError, ValueError, RecipeError) as error:
        print(f"prepare_train: {error}", file=sys.stderr)
        return 2
    print(f"prepare_train: {repo} at {revision} is in {found}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
