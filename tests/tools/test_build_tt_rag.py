"""Tests for tools/build_tt_rag.py: the fetch from GitHub, the tree checks, and where files may go.

Owner direction of 2026-10-23 (a retriever over the Tenstorrent
documentation). The contract these tests fix:

- fetch lists the pin's commit through the git trees API, downloads only the
  regular files the corpus spec selects (and its Doxyfile and the files that
  Doxyfile's INPUT lists), checks each against its git blob sha1, keeps a
  file already present with that blob, writes lassi-fetch.txt (the commit,
  then the corpus), and refuses a truncated listing, a download with another
  blob, and a destination inside the repository.
- build refuses, with status 2, a tree that is neither an install whose
  lassi-install.txt names the pin nor a fetch of the pin's commit for the
  corpus, an index path inside the repository, and a missing LASSI_SCRATCH
  when no --index is given; the default index is
  $LASSI_SCRATCH/rag/<corpus>@<VERSION>/index.json.
- check with no --index and no LASSI_SCRATCH says that no index file was
  named and checks the manifest only.
- the tree walk never follows a symbolic link.

Every tree, listing, and file here is SYNTHETIC, served by a fake opener; no
test opens a socket. No value here is a measurement.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path
from types import ModuleType

import pytest

from lassi.prompts import rag_corpus
from lassi.toolchains.pins import read_pin

REPO = Path(__file__).resolve().parents[2]
PIN = read_pin("tt-metal")
SPEC = REPO / "assets" / "rag" / "tt-metal-docs" / "corpus.yaml"
REPO_NAME = "tenstorrent/tt-metal"


def tool() -> ModuleType:
    """Import tools/build_tt_rag.py by path and return the module."""
    spec = importlib.util.spec_from_file_location("build_tt_rag", REPO / "tools" / "build_tt_rag.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def blob(data: bytes) -> str:
    """Return the git blob sha1 of `data`, computed here independently."""
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\x00" + data).hexdigest()


# SYNTHETIC upstream files at the pinned commit: three the spec's globs select, two its Doxyfile adds (one under a
# top-level directory no glob names), a stale Doxyfile entry, and two files nothing selects.
FILES = {
    "METALIUM_GUIDE.md": b"# Synthetic guide\n\nCircular buffers connect kernels.\n",
    "docs/Doxyfile": b"PROJECT_NAME = synthetic\nINPUT = tt_metal/hw/inc/api/dataflow/dataflow_api.h \\\n"
                     b"        tt_stl/span.hpp \\\n        tt_metal/include/removed.hpp\n",
    "docs/source/page.rst": b"Page\n====\n\nSynthetic prose.\n",
    "docs/source/picture.png": b"\x89PNG not selected",
    "tt_metal/hw/inc/api/dataflow/dataflow_api.h": b"// Reads over the NoC.\nvoid noc_async_read();\n",
    "tt_metal/hw/inc/api/dataflow/unlisted.h": b"void unlisted();\n",
    "tt_metal/api/tt-metalium/host_api.hpp": b"// Creates a program.\nProgram CreateProgram();\n",
    "tt_stl/span.hpp": b"// A span of elements.\ntemplate <class T> class Span;\n",
}
SELECTED = {"METALIUM_GUIDE.md", "docs/source/page.rst", "tt_metal/hw/inc/api/dataflow/dataflow_api.h",
            "tt_metal/api/tt-metalium/host_api.hpp", "tt_stl/span.hpp"}


class FakeGitHub:
    """Serves the trees API and raw files for FILES at the pin's commit, and records every URL asked for."""

    def __init__(self, files: dict[str, bytes], *, truncated: bool = False, corrupt: str | None = None) -> None:
        self.files, self.truncated, self.corrupt, self.urls = files, truncated, corrupt, []

    def __call__(self, url: str) -> bytes:
        self.urls.append(url)
        api = f"https://api.github.com/repos/{REPO_NAME}/git/trees/"
        raw = f"https://raw.githubusercontent.com/{REPO_NAME}/{PIN['COMMIT']}/"
        if url.startswith(api):
            sha = url[len(api):].split("?")[0]
            return json.dumps({"tree": self.listing(sha), "truncated": self.truncated}).encode()
        if url.startswith(raw):
            path = url[len(raw):]
            return b"tampered" if path == self.corrupt else self.files[path]
        raise AssertionError(f"unexpected URL {url}")

    def listing(self, sha: str) -> list[dict]:
        """Return the root listing for the commit, else the recursive listing of the top directory named `sha`."""
        if sha == PIN["COMMIT"]:
            tops = sorted({path.split("/")[0] for path in self.files})
            entries = [{"path": top, "type": "tree", "mode": "040000", "sha": f"tree-{top}"}
                       for top in tops if any(p.startswith(f"{top}/") for p in self.files)]
            entries += [{"path": p, "type": "blob", "mode": "100644", "sha": blob(d)}
                        for p, d in self.files.items() if "/" not in p]
            entries.append({"path": "submodule", "type": "commit", "mode": "160000", "sha": "0" * 40})
            return entries
        top = sha.removeprefix("tree-")
        entries = [{"path": p[len(top) + 1:], "type": "blob", "mode": "100644", "sha": blob(d)}
                   for p, d in self.files.items() if p.startswith(f"{top}/")]
        entries.append({"path": "source/link.md", "type": "blob", "mode": "120000", "sha": blob(b"page.rst")})
        return entries


def test_git_blob_sha1_matches_git() -> None:
    module = tool()
    assert module.git_blob_sha1(b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"
    assert module.git_blob_sha1(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"


def test_fetch_downloads_the_selected_files_checked_against_their_blobs(tmp_path: Path) -> None:
    module = tool()
    github = FakeGitHub(FILES)
    spec = rag_corpus.load_corpus_spec(SPEC)
    count = module.fetch(tmp_path / "tree", spec, PIN, opener=github)
    expected = SELECTED | {"docs/Doxyfile"}
    written = {p.relative_to(tmp_path / "tree").as_posix() for p in (tmp_path / "tree").rglob("*") if p.is_file()}
    assert written == expected | {"lassi-fetch.txt"} and count == len(expected)
    for path in expected:
        assert (tmp_path / "tree" / path).read_bytes() == FILES[path]
    record = (tmp_path / "tree" / "lassi-fetch.txt").read_text(encoding="ascii").splitlines()
    assert record[0] == PIN["COMMIT"] and record[1].startswith("corpus tt-metal-docs: ")
    assert not any("link.md" in url or "picture.png" in url or "unlisted.h" in url for url in github.urls)


def test_a_second_fetch_keeps_the_files_already_there(tmp_path: Path) -> None:
    module = tool()
    spec = rag_corpus.load_corpus_spec(SPEC)
    module.fetch(tmp_path, spec, PIN, opener=FakeGitHub(FILES))
    again = FakeGitHub(FILES)
    module.fetch(tmp_path, spec, PIN, opener=again)
    assert all("raw.githubusercontent.com" not in url for url in again.urls)


def test_fetch_refuses_a_download_with_another_blob(tmp_path: Path) -> None:
    module = tool()
    spec = rag_corpus.load_corpus_spec(SPEC)
    with pytest.raises(ValueError, match="docs/source/page.rst at .* downloaded with git blob"):
        module.fetch(tmp_path, spec, PIN, opener=FakeGitHub(FILES, corrupt="docs/source/page.rst"))
    assert not (tmp_path / "docs" / "source" / "page.rst").exists()
    assert not (tmp_path / "lassi-fetch.txt").exists()


def test_fetch_refuses_a_truncated_listing(tmp_path: Path) -> None:
    module = tool()
    with pytest.raises(ValueError, match="truncated"):
        module.fetch(tmp_path, rag_corpus.load_corpus_spec(SPEC), PIN, opener=FakeGitHub(FILES, truncated=True))


def test_fetch_refuses_a_destination_inside_the_repository() -> None:
    module = tool()
    with pytest.raises(ValueError, match="outside the repository"):
        module.fetch(REPO / "build" / "rag-tree", rag_corpus.load_corpus_spec(SPEC), PIN, opener=FakeGitHub(FILES))


def write_tree(root: Path, record: str | None, name: str = "lassi-fetch.txt") -> Path:
    """Write the FILES tree under `root` with `record` as its `name` record (none when None)."""
    for path, data in FILES.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_bytes(data)
    if record is not None:
        (root / name).write_text(record, encoding="ascii")
    return root


def build_args(tree: Path, out: Path) -> list[str]:
    """Return build arguments for `tree` that write the index and manifest under `out`."""
    return ["build", "--tree", str(tree), "--index", str(out / "index.json"), "--manifest", str(out / "M.yaml")]


def test_build_accepts_an_install_record_naming_the_pin(tmp_path: Path) -> None:
    tree = write_tree(tmp_path / "tree", f"{PIN['NAME']} {PIN['COMMIT']} {PIN['URL']}\n", "lassi-install.txt")
    assert tool().main(build_args(tree, tmp_path / "out")) == 0
    manifest = rag_corpus.read_manifest(tmp_path / "out" / "M.yaml")
    listed = SELECTED & set(rag_corpus.load_corpus_spec(SPEC).exclude)
    assert {row["path"] for row in manifest["files"]} == SELECTED - listed
    assert {row["path"] for row in manifest["excluded"] if row["reason"].startswith("listed: ")} == listed


@pytest.mark.parametrize(("record", "name", "message"), [
    (None, None, "lassi-install.txt cannot be read.*build_tt_rag.py fetch"),
    ("0" * 40 + "\ncorpus tt-metal-docs: x\n", "lassi-fetch.txt", "does not name commit"),
    (PIN["COMMIT"] + "\ncorpus another-corpus: x\n", "lassi-fetch.txt", "and corpus tt-metal-docs"),
    (f"{PIN['NAME']} {'0' * 40}\n", "lassi-install.txt", "not tt-metal " + PIN["COMMIT"]),
])
def test_build_refuses_a_tree_that_is_not_the_pinned_one(tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                                         record: str | None, name: str | None, message: str) -> None:
    tree = write_tree(tmp_path / "tree", record, name or "unused")
    assert tool().main(build_args(tree, tmp_path / "out")) == 2
    assert re.search(message, capsys.readouterr().err)
    assert not (tmp_path / "out").exists()


def test_build_refuses_an_index_inside_the_repository(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    tree = write_tree(tmp_path / "tree", PIN["COMMIT"] + "\ncorpus tt-metal-docs: x\n")
    args = ["build", "--tree", str(tree), "--index", str(REPO / "build" / "index.json"),
            "--manifest", str(tmp_path / "M.yaml")]
    assert tool().main(args) == 2
    assert "the index must lie outside the repository" in capsys.readouterr().err
    assert not (tmp_path / "M.yaml").exists()


def test_the_default_index_lies_under_lassi_scratch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                    capsys: pytest.CaptureFixture[str]) -> None:
    module = tool()
    spec = rag_corpus.load_corpus_spec(SPEC)
    monkeypatch.setenv("LASSI_SCRATCH", str(tmp_path))
    assert module.default_index(spec, PIN) == tmp_path / "rag" / f"tt-metal-docs@{PIN['VERSION']}" / "index.json"
    monkeypatch.delenv("LASSI_SCRATCH")
    tree = write_tree(tmp_path / "tree", PIN["COMMIT"] + "\ncorpus tt-metal-docs: x\n")
    assert module.main(["build", "--tree", str(tree), "--manifest", str(tmp_path / "M.yaml")]) == 2
    assert "LASSI_SCRATCH is not set" in capsys.readouterr().err


def test_check_with_no_index_says_so_and_checks_the_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                             capsys: pytest.CaptureFixture[str]) -> None:
    tree = write_tree(tmp_path / "tree", PIN["COMMIT"] + "\ncorpus tt-metal-docs: x\n")
    module = tool()
    assert module.main(build_args(tree, tmp_path / "out")) == 0
    capsys.readouterr()
    monkeypatch.delenv("LASSI_SCRATCH", raising=False)
    assert module.main(["check", "--tree", str(tree), "--manifest", str(tmp_path / "out" / "M.yaml")]) == 0
    out = capsys.readouterr().out
    assert "no index file named (pass --index <file>, or set LASSI_SCRATCH" in out
    assert "checked the manifest only" in out and "None" not in out


def test_the_tree_walk_never_follows_a_symbolic_link(tmp_path: Path) -> None:
    tree = write_tree(tmp_path / "tree", None)
    outside = tmp_path / "outside"
    (outside / "nested").mkdir(parents=True)
    (outside / "nested" / "secret.md").write_text("# Not in the tree\n", encoding="utf-8")
    (outside / "lone.rst").write_text("Lone\n====\n", encoding="utf-8")
    try:
        os.symlink(outside / "nested", tree / "docs" / "linked", target_is_directory=True)
        os.symlink(outside / "lone.rst", tree / "docs" / "lone.rst")
    except OSError as error:
        pytest.skip(f"this host cannot make symbolic links ({error})")
    spec = rag_corpus.load_corpus_spec(SPEC)
    selected = tool().tree_files(tree, spec)
    assert "docs/linked/secret.md" not in selected and "docs/lone.rst" not in selected
    assert "docs/source/page.rst" in selected

