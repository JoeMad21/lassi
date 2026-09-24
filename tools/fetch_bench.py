"""Materialize a suite's pinned sources under $LASSI_SCRATCH (task P0.8; bible Benchmark Suites).

Usage (on the build host, through tools/rx.py):

    uv run tools/fetch_bench.py assets/bench/lassi-hecbench-10.yaml

It checks out only the item directories the manifest lists, at the pinned
commit, into $LASSI_SCRATCH/bench/<suite>@<commit> (a sparse, shallow git
checkout), then verifies the commit and that every listed file exists. It is
idempotent: a checkout already at the commit with every file present is kept.
Sources never enter this repository.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from lassi.bench import Suite, load_suite, sources_dir  # noqa: E402


def sparse_dirs(suite: Suite) -> list[str]:
    """Return every item directory the manifest lists, sorted and without duplicates."""
    return sorted({spec.dir for item in suite.items.values() for spec in item.languages.values()})


def missing_files(suite: Suite, dest: Path) -> list[str]:
    """Return the listed files that do not exist under `dest`."""
    return [
        f"{spec.dir}/{file}"
        for item in suite.items.values()
        for spec in item.languages.values()
        for file in spec.files
        if not (dest / spec.dir / file).is_file()
    ]


def _git(dest: Path, *args: str) -> str:
    """Run git in `dest` and return its stdout; a failure raises CalledProcessError."""
    done = subprocess.run(["git", "-C", str(dest), *args], check=True, capture_output=True, text=True)
    return done.stdout.strip()


def _head(dest: Path) -> str:
    """Return the checked-out commit of `dest`, or '' when there is none."""
    try:
        return _git(dest, "rev-parse", "HEAD")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def fetch(suite: Suite, dest: Path) -> None:
    """Check out the suite's item directories at its commit into `dest` (sparse and shallow)."""
    dest.mkdir(parents=True, exist_ok=True)
    if not (dest / ".git").is_dir():
        _git(dest, "init", "--quiet")
        _git(dest, "remote", "add", "origin", suite.repo)
    _git(dest, "sparse-checkout", "set", "--no-cone", *[f"/{path}/" for path in sparse_dirs(suite)])
    _git(dest, "fetch", "--quiet", "--depth", "1", "origin", suite.commit)
    _git(dest, "checkout", "--quiet", "--detach", suite.commit)


def main(argv: list[str]) -> int:
    """Fetch the manifest's sources; print what was done and return a process status."""
    if len(argv) != 1:
        print("usage: fetch_bench.py <suite manifest>", file=sys.stderr)
        return 2
    scratch = os.environ.get("LASSI_SCRATCH", "")
    if not scratch:
        print("fetch_bench: LASSI_SCRATCH is not set; run this on the build host through tools/rx.py", file=sys.stderr)
        return 2
    suite = load_suite(Path(argv[0]))
    dest = sources_dir(Path(scratch), suite)
    if _head(dest) == suite.commit and not missing_files(suite, dest):
        print(f"fetch_bench: {dest} already holds {suite.name} at {suite.commit}")
        return 0
    fetch(suite, dest)
    head, missing = _head(dest), missing_files(suite, dest)
    if head != suite.commit or missing:
        print(f"fetch_bench: {dest} is at {head or 'nothing'}; missing files: {missing}", file=sys.stderr)
        return 1
    print(f"fetch_bench: fetched {suite.name} at {suite.commit} into {dest}")
    for path in sparse_dirs(suite):
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
