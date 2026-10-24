"""Tests for the content digest of a directory tree, the pin of a local checkpoint (2026-10-23; Agent Rule 10).

The contract (lassi.core.digest):

- tree_files(root) returns {relative POSIX path: sha256 of the file's bytes}
  for every regular file under `root`, in sorted path order, hidden files
  and files in subdirectories included.
- tree_sha256(root) is the sha256 of the listing text: one line
  "<file sha256>  <relative path>\\n" per file in that order, as sha256sum
  prints it, so the digest changes when a file's bytes, name, or place
  change, and never with the order files were written or their times.
- A root that is not a directory, a directory that holds no file, and a
  symbolic link anywhere under the root are ValueErrors naming the path.

Every byte here is SYNTHETIC; no value is a measurement.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from lassi.core import digest


def write(root: Path, files: dict[str, bytes]) -> Path:
    """Write `files` (relative path -> bytes) under `root` and return it."""
    for name, data in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return root


FILES = {"config.json": b"{}\n", "sub/weights.bin": b"\x00\x01SYNTHETIC", ".hidden": b"x"}


def test_tree_files_lists_every_regular_file_by_relative_path(tmp_path: Path) -> None:
    root = write(tmp_path / "ckpt", FILES)
    found = digest.tree_files(root)
    assert list(found) == sorted(FILES)
    assert found == {name: hashlib.sha256(data).hexdigest() for name, data in FILES.items()}


def test_tree_sha256_is_the_sha256_of_the_sha256sum_listing(tmp_path: Path) -> None:
    root = write(tmp_path / "ckpt", FILES)
    listing = "".join(f"{hashlib.sha256(FILES[name]).hexdigest()}  {name}\n" for name in sorted(FILES))
    assert digest.tree_sha256(root) == hashlib.sha256(listing.encode("ascii")).hexdigest()


def test_the_digest_ignores_write_order_and_times(tmp_path: Path) -> None:
    first = write(tmp_path / "a", FILES)
    second = write(tmp_path / "b", dict(reversed(list(FILES.items()))))
    os.utime(second / "config.json", (1, 1))
    assert digest.tree_sha256(first) == digest.tree_sha256(second)


@pytest.mark.parametrize("change", ["bytes", "name", "place", "extra"])
def test_the_digest_changes_with_any_file(change: str, tmp_path: Path) -> None:
    base = digest.tree_sha256(write(tmp_path / "base", FILES))
    changed = dict(FILES)
    if change == "bytes":
        changed["config.json"] = b"{ }\n"
    elif change == "name":
        changed["config2.json"] = changed.pop("config.json")
    elif change == "place":
        changed["sub/config.json"] = changed.pop("config.json")
    else:
        changed["extra.txt"] = b""
    assert digest.tree_sha256(write(tmp_path / "changed", changed)) != base


def test_a_missing_root_a_file_and_an_empty_directory_are_refused(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    (tmp_path / "file.txt").write_bytes(b"x")
    for path in (tmp_path / "absent", tmp_path / "file.txt", tmp_path / "empty"):
        with pytest.raises(ValueError, match=path.name):
            digest.tree_sha256(path)


def test_a_symbolic_link_is_refused(tmp_path: Path) -> None:
    root = write(tmp_path / "ckpt", FILES)
    try:
        (root / "link.json").symlink_to(root / "config.json")
    except OSError:
        pytest.skip("this host does not let the test create a symbolic link")
    with pytest.raises(ValueError, match="link.json"):
        digest.tree_sha256(root)
