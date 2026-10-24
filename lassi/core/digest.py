"""The content digest of a directory tree: how a local checkpoint directory is pinned (Agent Rule 10).

A trained checkpoint is a directory, not a Hub commit, so its pin is a
digest of its content. tree_files lists every regular file under the root
with the sha256 of its bytes, by relative POSIX path in sorted order, hidden
files and subdirectories included; tree_sha256 is the sha256 of that listing
written as sha256sum prints it, one line "<sha256>  <path>" with LF per
file. So the digest follows every file's bytes, name, and place, and never
the order the files were written, their times, or their modes. A root that
is not a directory, a tree with no file, and a symbolic link anywhere under
the root raise ValueError: a link's target could change after the digest is
taken. lassi train records each checkpoint's digest (lassi.train.run), and
hf_local checks it before it loads a checkpoint (lassi.llm.hf_local).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

# The block size files are hashed in, so a multi-gigabyte weights file is never read whole into memory.
_BLOCK = 1 << 20


def file_sha256(path: Path) -> str:
    """Return the sha256 hex digest of one file's bytes, read in blocks."""
    found = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(_BLOCK), b""):
            found.update(block)
    return found.hexdigest()


def tree_files(root: Path) -> dict[str, str]:
    """Return {relative POSIX path: sha256} of every regular file under `root`, in sorted path order.

    Raises ValueError for a root that is not a directory, a symbolic link
    under it, a file that is not regular, or a tree that holds no file.
    """
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"{root} is not a directory, so it has no content digest")
    found: dict[str, str] = {}
    for directory, names, files in os.walk(root, followlinks=False):
        for name in names + files:
            path = Path(directory) / name
            if path.is_symlink():
                raise ValueError(f"{path} is a symbolic link; a pinned tree holds regular files only")
        for name in files:
            path = Path(directory) / name
            if not path.is_file():
                raise ValueError(f"{path} is not a regular file; a pinned tree holds regular files only")
            found[path.relative_to(root).as_posix()] = file_sha256(path)
    if not found:
        raise ValueError(f"{root} holds no file, so it has no content digest")
    return dict(sorted(found.items()))


def tree_sha256(root: Path) -> str:
    """Return the sha256 of the sha256sum listing of `root`: "<sha256>  <path>\\n" per file in path order."""
    listing = "".join(f"{sha}  {path}\n" for path, sha in tree_files(root).items())
    return hashlib.sha256(listing.encode("utf-8")).hexdigest()
