"""Materialize a suite's pinned sources under $LASSI_SCRATCH (task P0.8; P4.13; bible Benchmark Suites).

Usage (on the build host, through tools/rx.py):

    uv run tools/fetch_bench.py assets/bench/<suite>.yaml

It places every file the manifest names in the pinned sources (each item's
files of every language that is not tracked, the item-level support files,
and each language's upstream support files) at the pinned commit in
$LASSI_SCRATCH/bench/<suite>@<commit>, then verifies the commit, that every
listed file exists, and that every file with a sha256 in the manifest has
that digest; a file that differs is named on stderr and the status is 1.
Tracked languages and tracked support files are in this repository and are
never fetched. Sources never enter this repository.

Two routes, chosen per suite (task P4.13):

- Copy from the installed pinned tree. A suite that names its source's pin
  (`pin: <name>`, such as tt-pairs-v0's `pin: tt-metal`) is copied file by
  file from $LASSI_TOOLCHAINS/<the pin's PREFIX_NAME> when that tree's
  lassi-install.txt names the pin's NAME and the manifest's commit on its
  first line (the install script writes it only after every check of the
  install passed), and the copy writes lassi-fetch.txt, whose first line is
  the commit and whose second names the tree. It needs no network and only
  reads the tree. load_suite has already checked that the pin's COMMIT and
  URL are the manifest's commit and repo.
- Otherwise a sparse, shallow git checkout of the pinned commit, fetched by
  sha, holding only the listed files and item directories.

It is idempotent: a destination already at the commit (its git HEAD, or the
first line of its lassi-fetch.txt) with every file present and matching is
kept, and nothing is copied or fetched.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from lassi.bench import Suite, load_suite, sources_dir  # noqa: E402
from lassi.toolchains import pins  # noqa: E402

# The record a copy from an installed tree leaves in the destination: the commit, then the tree.
FETCH_RECORD = "lassi-fetch.txt"
# The record an installed toolchain tree holds once every check of its install passed.
INSTALL_RECORD = "lassi-install.txt"


def _item_dirs(suite: Suite) -> set[str]:
    """Return every item directory of a language that is not tracked."""
    return {spec.dir for item in suite.items.values() for spec in item.languages.values() if not spec.tracked}


def _support_paths(suite: Suite) -> list[str]:
    """Return every support path in the pinned sources: item-level, then each language's upstream ones, once each."""
    item_level = [path for item in suite.items.values() for path in item.support.values()]
    upstream = [
        entry.path
        for item in suite.items.values()
        for spec in item.languages.values()
        for entry in spec.support.values()
        if not entry.tracked
    ]
    return list(dict.fromkeys([*item_level, *upstream]))


def _outside(path: str, dirs: set[str]) -> bool:
    """Return True when `path` lies in none of `dirs`."""
    return not any(path.startswith(f"{directory}/") for directory in dirs)


def sparse_dirs(suite: Suite) -> list[str]:
    """Return what the sparse checkout holds, sorted: every untracked item directory, and each support file outside."""
    dirs = _item_dirs(suite)
    return sorted(dirs | {path for path in _support_paths(suite) if _outside(path, dirs)})


def listed_files(suite: Suite) -> list[str]:
    """Return every file the manifest names in the pinned sources: untracked item files, then the support files."""
    files = [
        f"{spec.dir}/{file}"
        for item in suite.items.values()
        for spec in item.languages.values()
        if not spec.tracked
        for file in spec.files
    ]
    return list(dict.fromkeys([*files, *_support_paths(suite)]))


def missing_files(suite: Suite, dest: Path) -> list[str]:
    """Return the listed files, support files included, that do not exist under `dest`."""
    return [path for path in listed_files(suite) if not (dest / path).is_file()]


def _digests(suite: Suite) -> dict[str, str]:
    """Return every path in the pinned sources the manifest records a sha256 for, with that digest."""
    found: dict[str, str] = {}
    for item in suite.items.values():
        for spec in item.languages.values():
            found.update({f"{spec.dir}/{file}": digest for file, digest in spec.sha256.items()})
            found.update({entry.path: entry.sha256 for entry in spec.support.values() if entry.sha256})
    return found


def mismatched_files(suite: Suite, dest: Path) -> list[str]:
    """Return the files under `dest` whose sha256 differs from the one the manifest records for them."""
    return [
        path
        for path, digest in _digests(suite).items()
        if (dest / path).is_file() and hashlib.sha256((dest / path).read_bytes()).hexdigest() != digest
    ]


def _git(dest: Path, *args: str) -> str:
    """Run git in `dest` and return its stdout; a failure raises CalledProcessError."""
    done = subprocess.run(["git", "-C", str(dest), *args], check=True, capture_output=True, text=True)
    return done.stdout.strip()


def _head(dest: Path) -> str:
    """Return the checked-out commit of `dest`, or '' when it is no git checkout of its own.

    A directory without .git gives '' at once, so git never walks up to a
    repository that holds `dest`.
    """
    if not (dest / ".git").exists():
        return ""
    try:
        return _git(dest, "rev-parse", "HEAD")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def fetched_commit(dest: Path) -> str:
    """Return the commit `dest` holds: its git HEAD, else the first line of its lassi-fetch.txt, else ''."""
    head = _head(dest)
    if head:
        return head
    try:
        lines = (dest / FETCH_RECORD).read_bytes().decode("utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return ""
    return lines[0].strip() if lines else ""


def installed_tree(suite: Suite) -> Path | None:
    """Return the installed pinned tree to copy `suite` from, or None (the module docstring's copy route).

    It is $LASSI_TOOLCHAINS/<PREFIX_NAME of the suite's pin>, given only when
    its lassi-install.txt's first line names the pin's NAME and the suite's
    commit; a suite with no pin, no $LASSI_TOOLCHAINS, or no such record
    gives None.
    """
    root = os.environ.get("LASSI_TOOLCHAINS", "")
    if suite.pin is None or not root:
        return None
    pin = pins.read_pin(suite.pin)
    tree = Path(root) / pin.get("PREFIX_NAME", "")
    try:
        lines = (tree / INSTALL_RECORD).read_bytes().decode("utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return None
    words = lines[0].split()[:2] if lines else []
    return tree if words == [pin.get("NAME"), suite.commit] else None


def copy_from_tree(suite: Suite, tree: Path, dest: Path) -> None:
    """Copy every listed file from the installed tree into `dest` and write lassi-fetch.txt; the tree is only read.

    A listed file the tree lacks is left missing, for main to report.
    """
    dest.mkdir(parents=True, exist_ok=True)
    for path in listed_files(suite):
        source = tree / path
        if source.is_file():
            (dest / path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest / path)
    (dest / FETCH_RECORD).write_bytes(f"{suite.commit}\ncopied from {tree}\n".encode("utf-8"))


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
    if fetched_commit(dest) == suite.commit and not missing_files(suite, dest) and not mismatched_files(suite, dest):
        print(f"fetch_bench: {dest} already holds {suite.name} at {suite.commit}")
        return 0
    tree = installed_tree(suite)
    if tree is not None:
        copy_from_tree(suite, tree, dest)
    else:
        fetch(suite, dest)
    head, missing, mismatched = fetched_commit(dest), missing_files(suite, dest), mismatched_files(suite, dest)
    if head != suite.commit or missing or mismatched:
        print(
            f"fetch_bench: {dest} is at {head or 'nothing'}; missing files: {missing}; "
            f"files whose sha256 differs from the manifest: {mismatched}",
            file=sys.stderr,
        )
        return 1
    how = f"copied from {tree}" if tree is not None else "fetched"
    print(f"fetch_bench: {how}: {suite.name} at {suite.commit} into {dest}")
    for path in sparse_dirs(suite):
        print(f"  {path}")
    print(f"fetch_bench: {len(listed_files(suite))} files present, sha256 checked for {len(_digests(suite))}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
