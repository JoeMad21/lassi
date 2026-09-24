"""Materialize a suite's pinned sources under $LASSI_SCRATCH (task P0.8; bible Benchmark Suites).

Usage (on the build host, through tools/rx.py):

    uv run tools/fetch_bench.py assets/bench/lassi-hecbench-10.yaml

It checks out only the item directories and support files the manifest
lists, at the pinned commit, into $LASSI_SCRATCH/bench/<suite>@<commit> (a
sparse, shallow git checkout), then verifies the commit, that every listed
file exists, and that every file with a sha256 in the manifest has that
digest; a file that differs is named on stderr and the status is 1. It is
idempotent: a checkout already at the commit with every file present and
matching is kept. Sources never enter this repository.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from lassi.bench import Suite, load_suite, sources_dir  # noqa: E402


def _item_dirs(suite: Suite) -> set[str]:
    """Return every item directory the manifest lists."""
    return {spec.dir for item in suite.items.values() for spec in item.languages.values()}


def _support_paths(suite: Suite) -> list[str]:
    """Return every support file path the manifest lists, in manifest order and without duplicates."""
    return list(dict.fromkeys(path for item in suite.items.values() for path in item.support.values()))


def _outside(path: str, dirs: set[str]) -> bool:
    """Return True when `path` lies in none of `dirs`."""
    return not any(path.startswith(f"{directory}/") for directory in dirs)


def sparse_dirs(suite: Suite) -> list[str]:
    """Return what the sparse checkout holds, sorted: every item directory, and each support file outside them."""
    dirs = _item_dirs(suite)
    return sorted(dirs | {path for path in _support_paths(suite) if _outside(path, dirs)})


def listed_files(suite: Suite) -> list[str]:
    """Return every file the manifest lists: each item's files per language, then the support files not yet listed."""
    files = [
        f"{spec.dir}/{file}" for item in suite.items.values() for spec in item.languages.values() for file in spec.files
    ]
    return list(dict.fromkeys([*files, *_support_paths(suite)]))


def missing_files(suite: Suite, dest: Path) -> list[str]:
    """Return the listed files, support files included, that do not exist under `dest`."""
    return [path for path in listed_files(suite) if not (dest / path).is_file()]


def mismatched_files(suite: Suite, dest: Path) -> list[str]:
    """Return the files under `dest` whose sha256 differs from the one the manifest records for them."""
    return [
        f"{spec.dir}/{file}"
        for item in suite.items.values()
        for spec in item.languages.values()
        for file, digest in spec.sha256.items()
        if (dest / spec.dir / file).is_file()
        and hashlib.sha256((dest / spec.dir / file).read_bytes()).hexdigest() != digest
    ]


def _digest_count(suite: Suite) -> int:
    """Return how many files the manifest records a sha256 for."""
    return sum(len(spec.sha256) for item in suite.items.values() for spec in item.languages.values())


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
    """Check out the suite's item directories and support files at its commit into `dest` (sparse and shallow).

    Line endings are kept as committed (core.autocrlf off), so the files keep
    the bytes their sha256 was taken from, and the checkout is forced, so a
    changed tracked file gets its pinned bytes back.
    """
    dest.mkdir(parents=True, exist_ok=True)
    if not (dest / ".git").is_dir():
        _git(dest, "init", "--quiet")
        _git(dest, "remote", "add", "origin", suite.repo)
    _git(dest, "config", "core.autocrlf", "false")
    dirs = _item_dirs(suite)
    patterns = [f"/{path}/" if path in dirs else f"/{path}" for path in sparse_dirs(suite)]
    _git(dest, "sparse-checkout", "set", "--no-cone", *patterns)
    _git(dest, "fetch", "--quiet", "--depth", "1", "origin", suite.commit)
    _git(dest, "checkout", "--quiet", "--force", "--detach", suite.commit)


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
    if _head(dest) == suite.commit and not missing_files(suite, dest) and not mismatched_files(suite, dest):
        print(f"fetch_bench: {dest} already holds {suite.name} at {suite.commit}")
        return 0
    fetch(suite, dest)
    head, missing, mismatched = _head(dest), missing_files(suite, dest), mismatched_files(suite, dest)
    if head != suite.commit or missing or mismatched:
        print(
            f"fetch_bench: {dest} is at {head or 'nothing'}; missing files: {missing}; "
            f"files whose sha256 differs from the manifest: {mismatched}",
            file=sys.stderr,
        )
        return 1
    print(f"fetch_bench: fetched {suite.name} at {suite.commit} into {dest}")
    for path in sparse_dirs(suite):
        print(f"  {path}")
    print(f"fetch_bench: {len(listed_files(suite))} files present, sha256 checked for {_digest_count(suite)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
