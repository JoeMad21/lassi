"""Fetch upstream LASSI at its pinned commit into third_party/LASSI (task P1.1).

Usage:

    uv run python tools/fetch_upstream.py [MANIFEST] [--dest DIR]

MANIFEST (default assets/upstream/lassi.yaml) names the upstream `url`, the
pinned `commit` (a full 40-digit id), and the checkout `path` relative to the
repository root; DIR defaults to that path. The tool fetches exactly the
pinned commit (shallow, by id), detaches HEAD there, and checks that HEAD is
the pin and the tree is clean.

- A clean checkout already at the pin is left untouched and no network is
  used, so a rerun changes nothing.
- A clean checkout of the same repository at another commit is moved to the
  pin. A checkout whose .git is a file (the layout a submodule clone leaves)
  is accepted like any other.
- It refuses, and leaves exactly as found, a checkout with local edits or
  untracked files, a checkout of another repository (origin differs), and a
  non-empty directory that is not a checkout of its own.
- It never deletes a file upstream does not track, a repository, or its
  objects. A fetch that fails can leave an empty repository at DIR; the next
  run fetches into it. A fetch that runs past FETCH_TIMEOUT_S seconds is
  stopped and fails like any other failed fetch.

The checkout is gitignored and never tracked; nothing in it is edited or run
here. Exit status: 0 when DIR holds the pin, 1 on a refusal or a failed
fetch (the message names the pinned commit), 2 on a bad manifest.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import yaml

REPO = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO / "assets" / "upstream" / "lassi.yaml"
KEYS = ("url", "commit", "path")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
# A stalled network ends the fetch with a FetchError instead of a hang.
FETCH_TIMEOUT_S = 600
# Environment variables that would point git at a repository other than the checkout.
REPO_VARS = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
             "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_COMMON_DIR", "GIT_NAMESPACE")


class ManifestError(ValueError):
    """The pin manifest is missing, unreadable, or lacks a valid key."""


class Refusal(RuntimeError):
    """The destination cannot reach the pin without changing something the tool must keep."""


class FetchError(RuntimeError):
    """Git could not fetch or check out the pinned commit."""


@dataclass(frozen=True)
class Pin:
    """The pinned upstream: clone URL, full commit id, and checkout path relative to the repository root."""

    url: str
    commit: str
    path: str


def load_pin(path: Path) -> Pin:
    """Read and check the pin manifest at `path`; a missing or bad key raises ManifestError naming it."""
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as err:
        raise ManifestError(f"{path}: cannot read the manifest: {err}") from err
    if not isinstance(data, dict):
        raise ManifestError(f"{path}: the manifest is not a mapping with keys {', '.join(KEYS)}")
    for key in KEYS:
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ManifestError(f"{path}: missing key '{key}' (a non-empty string)")
    if not COMMIT_RE.fullmatch(data["commit"]):
        raise ManifestError(f"{path}: key 'commit' is not a full 40-digit commit id: {data['commit']!r}")
    rel = PurePosixPath(data["path"])
    if rel.is_absolute() or ".." in rel.parts or ":" in data["path"]:
        raise ManifestError(f"{path}: key 'path' is not a path inside the repository: {data['path']!r}")
    return Pin(url=data["url"].strip(), commit=data["commit"], path=rel.as_posix())


def _env(dest: Path) -> dict[str, str]:
    """Return an environment where git sees only `dest` (never a repository above it) and never prompts."""
    env = {key: value for key, value in os.environ.items() if key not in REPO_VARS}
    env["GIT_CEILING_DIRECTORIES"] = str(dest.parent)
    env["GIT_TERMINAL_PROMPT"] = "0"
    return env


def _git(dest: Path, *args: str, check: bool = True,
         timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    """Run git in `dest` without optional locks (a status never rewrites the index).

    Git is stopped after `timeout` seconds (None: no limit), and that raises FetchError.
    """
    try:
        done = subprocess.run(["git", "--no-optional-locks", "-C", str(dest), *args],
                              capture_output=True, text=True, env=_env(dest), timeout=timeout)
    except subprocess.TimeoutExpired as err:
        raise FetchError(f"git {' '.join(args)} timed out after {timeout} s in {dest}") from err
    if check and done.returncode != 0:
        detail = (done.stderr or done.stdout).strip().splitlines()
        raise FetchError(f"git {' '.join(args)} failed in {dest}: {detail[-1] if detail else done.returncode}")
    return done


def _head(dest: Path) -> str:
    """Return the full id of HEAD in `dest`, or '' when nothing is checked out."""
    done = _git(dest, "rev-parse", "--verify", "--quiet", "HEAD^{commit}", check=False)
    return done.stdout.strip() if done.returncode == 0 else ""


def _same_url(left: str, right: str) -> bool:
    """Return whether two clone URLs name the same repository (a trailing slash or .git aside)."""
    def norm(url: str) -> str:
        return url.strip().rstrip("/").removesuffix(".git")
    return norm(left) == norm(right)


def inspect(dest: Path, pin: Pin) -> str:
    """Return 'empty', 'at-pin', or 'move' for `dest`; raise Refusal when it must be left alone.

    Only read-only git commands run here, and only once `dest/.git` exists.
    """
    if not dest.exists() or (dest.is_dir() and not any(dest.iterdir())):
        return "empty"
    if not dest.is_dir() or not (dest / ".git").exists():
        raise Refusal(f"{dest} is not empty and is not a git checkout; move it aside to fetch the pin")
    top = _git(dest, "rev-parse", "--show-toplevel", check=False)
    if top.returncode != 0 or Path(top.stdout.strip()).resolve() != dest.resolve():
        raise Refusal(f"{dest} is not a git checkout of its own")
    origin = _git(dest, "remote", "get-url", "origin", check=False).stdout.strip()
    if not _same_url(origin, pin.url):
        raise Refusal(f"{dest} is a checkout of {origin or 'a repository without an origin'}, not {pin.url}")
    changes = _git(dest, "status", "--porcelain").stdout
    if changes:
        raise Refusal(f"{dest} has local changes, kept as they are:\n{changes.rstrip()}")
    if _head(dest) == pin.commit:
        return "at-pin"
    ignored = _git(dest, "status", "--porcelain", "--ignored").stdout
    if ignored:
        raise Refusal(f"{dest} holds ignored files a checkout could overwrite, kept as they are:\n{ignored.rstrip()}")
    return "move"


def _fetch(dest: Path, pin: Pin) -> None:
    """Fetch the pinned commit from origin into `dest` (shallow unless the repository is complete)."""
    shallow = not _head(dest) or _git(dest, "rev-parse", "--is-shallow-repository").stdout.strip() == "true"
    depth = ["--depth", "1"] if shallow else []
    done = _git(dest, "fetch", "--quiet", "--no-tags", *depth, "origin", pin.commit, check=False,
                timeout=FETCH_TIMEOUT_S)
    if done.returncode != 0:
        detail = (done.stderr or done.stdout).strip().splitlines()
        raise FetchError(f"cannot fetch {pin.commit} from {pin.url}: {detail[-1] if detail else done.returncode}")


def fetch_into_empty(dest: Path, pin: Pin) -> None:
    """Create a repository at `dest` whose origin is the pin's URL, fetch the pin, and detach HEAD there."""
    dest.mkdir(parents=True, exist_ok=True)
    _git(dest, "init", "--quiet")
    _git(dest, "config", "core.autocrlf", "false")  # upstream's bytes, whatever the system config says
    _git(dest, "remote", "add", "origin", pin.url)
    _fetch(dest, pin)
    _git(dest, "checkout", "--quiet", "--detach", pin.commit)


def move_to_pin(dest: Path, pin: Pin) -> None:
    """Detach HEAD of a clean checkout at the pin, fetching the commit first when it is absent."""
    if _git(dest, "cat-file", "-e", f"{pin.commit}^{{commit}}", check=False).returncode != 0:
        _fetch(dest, pin)
    _git(dest, "checkout", "--quiet", "--detach", pin.commit)


def verify(dest: Path, pin: Pin) -> None:
    """Raise FetchError unless `dest` is its own clean checkout with HEAD at the pin."""
    head = _head(dest)
    if head != pin.commit:
        raise FetchError(f"{dest} is at {head or 'no commit'} after the fetch, not {pin.commit}")
    changes = _git(dest, "status", "--porcelain").stdout
    if changes:
        raise FetchError(f"{dest} is not clean after the checkout:\n{changes.rstrip()}")


def _shown(dest: Path) -> str:
    """Return `dest` relative to the repository root when it lies inside it, else as given."""
    try:
        return dest.relative_to(REPO).as_posix()
    except ValueError:
        return str(dest)


def main(argv: list[str] | None = None) -> int:
    """Bring the checkout to the manifest's pin; print what was done and return a process status."""
    parser = argparse.ArgumentParser(prog="fetch_upstream.py", description=__doc__.splitlines()[0])
    parser.add_argument("manifest", nargs="?", type=Path, default=DEFAULT_MANIFEST, help="pin manifest")
    parser.add_argument("--dest", type=Path, help="checkout directory (default: the manifest's path)")
    args = parser.parse_args(argv)
    try:
        pin = load_pin(args.manifest)
    except ManifestError as err:
        print(f"fetch_upstream: {err}", file=sys.stderr)
        return 2
    dest = (args.dest if args.dest is not None else REPO / pin.path).resolve()
    try:
        state = inspect(dest, pin)
        if state == "at-pin":
            print(f"fetch_upstream: {_shown(dest)} already holds {pin.url} at {pin.commit}")
            return 0
        (fetch_into_empty if state == "empty" else move_to_pin)(dest, pin)
        verify(dest, pin)
    except (Refusal, FetchError, OSError) as err:
        print(f"fetch_upstream: {err}\nfetch_upstream: {_shown(dest)} does not hold the pin {pin.commit}",
              file=sys.stderr)
        return 1
    done = "fetched" if state == "empty" else "moved"
    print(f"fetch_upstream: {done} {pin.url} at {pin.commit} into {_shown(dest)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
