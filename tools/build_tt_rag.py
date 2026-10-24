"""Build the Tenstorrent documentation index from the pinned tt-metal tree (owner direction of 2026-10-23).

Usage:

    uv run tools/build_tt_rag.py fetch --dest <dir>
    uv run tools/build_tt_rag.py build --tree <dir> [--index <file>] [--corpus <spec>] [--manifest <file>]
    uv run tools/build_tt_rag.py check --tree <dir> [--index <file>] [--corpus <spec>] [--manifest <file>]

The corpus spec (default assets/rag/tt-metal-docs/corpus.yaml) names the pin,
the files, and the Agent Rule 5 exclusions (lassi.prompts.rag_corpus). The
tree is the pinned tt-metal at the pin's COMMIT, either an install
($LASSI_TOOLCHAINS/<PREFIX_NAME>, whose lassi-install.txt names the pin's
NAME and COMMIT on its first line) or a directory `fetch` filled, whose
lassi-fetch.txt names the commit and the corpus. Any other tree is refused.

- fetch downloads from GitHub, at exactly the pin's commit, the files the
  spec selects (and the Doxyfile it names), listed through the git trees
  API, each checked against its git blob sha1, into `--dest`, which must lie
  outside this repository. A file already there with that blob is kept.
- build writes the index (lassi.prompts.retrieval index_bytes) to `--index`,
  by default $LASSI_SCRATCH/rag/<corpus>@<pin VERSION>/index.json (never in
  this repository), and the manifest (default MANIFEST.yaml beside the spec),
  which records the source pin, every file's sha256, the parameters, every
  excluded file and excised line range, the similarity report (each indexed
  code file at or above the spec's report_at, with its closest Tier A
  reference; printed too, never a refusal), and the index's sha256.
- check builds in memory and compares with the manifest, and with the index
  file when one is given or found at the default; status 1 names each
  difference, and nothing is written. With no --index and no LASSI_SCRATCH
  it says so and checks the manifest only.

The same tree and spec always give the same index bytes and manifest text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from lassi.bench import load_suite  # noqa: E402
from lassi.prompts import rag_corpus  # noqa: E402
from lassi.prompts.retrieval import index_bytes  # noqa: E402
from lassi.toolchains import pins  # noqa: E402
from lassi.toolchains.ttmetal_build import check_install_record  # noqa: E402

DEFAULT_CORPUS = REPO / "assets" / "rag" / "tt-metal-docs" / "corpus.yaml"
# The record `fetch` leaves in its destination: the commit, then the corpus it was fetched for.
FETCH_RECORD = "lassi-fetch.txt"
INDEX_NAME = "index.json"
API_TREE = "https://api.github.com/repos/{repo}/git/trees/{sha}"
RAW_FILE = "https://raw.githubusercontent.com/{repo}/{commit}/{path}"
# The git modes of regular files; symbolic links (120000) and submodules (160000) are never corpus files.
FILE_MODES = ("100644", "100755")
TIMEOUT_S = 60
WORKERS = 8
_GLOB = set("*?[")

Opener = Callable[[str], bytes]


def main(argv: list[str] | None = None) -> int:
    """Run one subcommand; return 0 on success, 1 when check finds a difference, 2 on a refusal."""
    args = _parser().parse_args(argv)
    try:
        spec_path = Path(args.corpus).resolve()
        spec = rag_corpus.load_corpus_spec(spec_path)
        pin = pins.read_pin(spec.pin)
        if args.command == "fetch":
            count = fetch(Path(args.dest), spec, pin)
            print(f"build_tt_rag: {count} files of {pin['NAME']} at {pin['COMMIT']} in {args.dest}")
            return 0
        manifest_path = Path(args.manifest) if args.manifest else spec_path.parent / "MANIFEST.yaml"
        if args.index:
            index_path: Path | None = Path(args.index)
        elif args.command == "check" and not os.environ.get("LASSI_SCRATCH"):
            index_path = None
        else:
            index_path = default_index(spec, pin)
        if args.command == "build" and index_path is not None:
            _refuse_inside_repo(index_path, "the index")
        document, data = build(Path(args.tree), spec_path, spec, pin)
    except (OSError, ValueError) as error:
        print(f"build_tt_rag: {error}", file=sys.stderr)
        return 2
    if args.command == "check" or index_path is None:
        return check(document, manifest_path, index_path)
    _write(index_path, data)
    _write(manifest_path, rag_corpus.manifest_text(document).encode("ascii"))
    index = document["index"]
    print(f"build_tt_rag: {index['files']} files, {index['chunks']} chunks, {len(document['excluded'])} excluded, "
          f"{len(document['excised'])} line ranges excised; index {index_path} sha256 {index['sha256']}")
    for row in document["similar"]:
        print(f"build_tt_rag: similar: {row['path']} has token similarity {row['similarity']} to {row['reference']}")
    return 0


def build(tree: Path, spec_path: Path, spec: rag_corpus.CorpusSpec,
          pin: Mapping[str, str]) -> tuple[dict[str, Any], bytes]:
    """Return (manifest document, index bytes) for `tree`; refuse a tree that is not the pinned one."""
    check_tree(tree, pin, spec.corpus)
    suite = load_suite(REPO / spec.suite)
    if suite.commit != pin["COMMIT"]:
        raise ValueError(f"{spec.suite} pins {suite.commit}, not {spec.pin}'s COMMIT {pin['COMMIT']}")
    tier_dirs = rag_corpus.tier_a_dirs(suite, spec.examples)
    tracked = rag_corpus.tracked_references(suite, REPO)
    built = rag_corpus.build_corpus(tree_files(tree, spec), spec, tier_dirs=tier_dirs, references=tracked)
    spec_sha256 = hashlib.sha256(spec_path.read_bytes()).hexdigest()
    source = {"pin": spec.pin, "repo": pin["URL"], "commit": pin["COMMIT"]}
    meta = {
        "corpus": spec.corpus,
        "corpus_sha256": spec_sha256,
        "source": {"repo": pin["URL"], "commit": pin["COMMIT"]},
        "chunking": {"max_lines": spec.max_lines, "max_chars": spec.max_chars, "min_piece_lines": spec.min_piece_lines},
        "bm25": {"k1": spec.k1, "b": spec.b},
    }
    data = index_bytes(built.chunks, meta)
    document = rag_corpus.manifest_document(
        built, spec, spec_sha256=spec_sha256, source=source, tier_dirs=tier_dirs, tracked=tracked,
        index_sha256=hashlib.sha256(data).hexdigest(), index_size=len(data))
    return document, data


def check(document: Mapping[str, Any], manifest_path: Path, index_path: Path | None) -> int:
    """Return 0 when the manifest (and the index file, when one exists) equal this build; else name each difference."""
    problems: list[str] = []
    try:
        recorded = rag_corpus.read_manifest(manifest_path)
    except ValueError as error:
        recorded, problems = {}, [str(error)]
    for key in sorted(rag_corpus.MANIFEST_KEYS):
        if recorded and recorded.get(key) != json.loads(json.dumps(document[key])):
            problems.append(f"{manifest_path}: {key} differs from this build")
    if index_path is None:
        print("build_tt_rag: no index file named (pass --index <file>, or set LASSI_SCRATCH for the default); "
              "checked the manifest only")
    elif index_path.is_file():
        digest = hashlib.sha256(index_path.read_bytes()).hexdigest()
        if digest != document["index"]["sha256"]:
            problems.append(f"{index_path} has sha256 {digest}, this build {document['index']['sha256']}")
    else:
        print(f"build_tt_rag: no index at {index_path}; checked the manifest only")
    for problem in problems:
        print(f"build_tt_rag: {problem}", file=sys.stderr)
    if not problems:
        print(f"build_tt_rag: {manifest_path} matches this build (index sha256 {document['index']['sha256']})")
    return 1 if problems else 0


def check_tree(tree: Path, pin: Mapping[str, str], corpus: str) -> None:
    """Raise ValueError unless `tree` is an install of the pin or a fetch of the pin's commit for `corpus`."""
    record = tree / FETCH_RECORD
    if record.is_file():
        lines = record.read_bytes().decode("utf-8", errors="replace").splitlines()
        if lines[:1] == [pin["COMMIT"]] and len(lines) > 1 and lines[1].startswith(f"corpus {corpus}:"):
            return
        raise ValueError(f"{record} does not name commit {pin['COMMIT']} and corpus {corpus}; fetch the tree "
                         f"again with uv run tools/build_tt_rag.py fetch")
    try:
        check_install_record(tree, pin)
    except ValueError as error:
        raise ValueError(f"{error}; or fetch the files with uv run tools/build_tt_rag.py fetch") from None


def tree_files(tree: Path, spec: rag_corpus.CorpusSpec) -> dict[str, bytes]:
    """Return {path: bytes} of every regular file of `tree` the spec selects: its globs and its Doxyfile's INPUT.

    Symbolic links are never followed or read, as fetch never lists them, so
    an install and a fetch of one commit select the same files.
    """
    selected: dict[str, bytes] = {}
    for root in sorted({_literal_root(pattern) for pattern in spec.include}):
        for relative in _regular_files(tree, root):
            if any(rag_corpus.glob_match(pattern, relative) for pattern in spec.include):
                selected[relative] = (tree / relative).read_bytes()
    for relative in doxygen_paths(tree, spec):
        if _is_regular(tree, relative):
            selected[relative] = (tree / relative).read_bytes()
    return dict(sorted(selected.items()))


def _regular_files(tree: Path, root: str) -> list[str]:
    """Return the tree paths of the regular files at or under `root`, never through a symbolic link."""
    if _is_regular(tree, root):
        return [root]
    base = tree / root if root else tree
    if not base.is_dir() or any((tree / part).is_symlink() for part in _prefixes(root)):
        return []
    found: list[str] = []
    for directory, _, names in os.walk(base, followlinks=False):
        for name in names:
            path = Path(directory) / name
            if not path.is_symlink() and path.is_file():
                found.append(path.relative_to(tree).as_posix())
    return sorted(found)


def _is_regular(tree: Path, relative: str) -> bool:
    """Return True when `relative` is a regular file of `tree` reached through no symbolic link."""
    if not relative:
        return False
    return (tree / relative).is_file() and not any((tree / part).is_symlink() for part in _prefixes(relative))


def _prefixes(relative: str) -> list[str]:
    """Return every leading part of a POSIX path, itself included: a/b/c gives a, a/b, a/b/c."""
    parts = [part for part in relative.split("/") if part]
    return ["/".join(parts[:count]) for count in range(1, len(parts) + 1)]


def doxygen_paths(tree: Path, spec: rag_corpus.CorpusSpec) -> list[str]:
    """Return the INPUT paths of the spec's Doxyfile in `tree` ([] when the spec names none)."""
    if spec.doxygen is None:
        return []
    text = (tree / spec.doxygen).read_bytes().decode("utf-8", errors="replace")
    return [path.strip("/") for path in rag_corpus.doxygen_inputs(text)]


def fetch(dest: Path, spec: rag_corpus.CorpusSpec, pin: Mapping[str, str], opener: Opener | None = None) -> int:
    """Fill `dest` with the spec's files at the pin's commit from GitHub; return how many files it holds.

    Every file is checked against the blob sha1 of the commit's git tree, so
    the bytes are the commit's exactly. `opener` (default: urllib) returns a
    URL's bytes; tests pass their own.
    """
    _refuse_inside_repo(dest, "the fetch destination")
    dest.mkdir(parents=True, exist_ok=True)
    get = opener or _urlopen
    repo, commit = _github_repo(pin["URL"]), pin["COMMIT"]
    root = _tree(get, repo, commit, recursive=False)
    tops = {pattern.split("/")[0] for pattern in spec.include}
    if spec.doxygen is not None:
        tops.add(spec.doxygen.split("/")[0])
    blobs = _listing(get, repo, root, tops)
    wanted: dict[str, str] = {}
    if spec.doxygen is not None:
        if spec.doxygen not in blobs:
            raise ValueError(f"{spec.doxygen} is not a file of {repo} at {commit}")
        _fetch_one(get, repo, commit, dest, spec.doxygen, blobs[spec.doxygen])
        inputs = doxygen_paths(dest, spec)
        blobs.update(_listing(get, repo, root, {path.split("/")[0] for path in inputs} - tops))
        wanted.update({path: blobs[path] for path in inputs if path in blobs})
    wanted.update({path: sha for path, sha in blobs.items()
                   if any(rag_corpus.glob_match(pattern, path) for pattern in spec.include)})
    with ThreadPoolExecutor(WORKERS) as pool:
        list(pool.map(lambda item: _fetch_one(get, repo, commit, dest, *item), sorted(wanted.items())))
    record = f"{commit}\ncorpus {spec.corpus}: files of {pin['URL']} at {commit}, each checked against its git blob\n"
    (dest / FETCH_RECORD).write_bytes(record.encode("ascii"))
    return len(set(wanted) | ({spec.doxygen} if spec.doxygen is not None else set()))


def git_blob_sha1(data: bytes) -> str:
    """Return the git blob id of `data`: the sha1 of b'blob <size>\\0' followed by the bytes."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def default_index(spec: rag_corpus.CorpusSpec, pin: Mapping[str, str]) -> Path:
    """Return $LASSI_SCRATCH/rag/<corpus>@<VERSION>/index.json; raise ValueError when LASSI_SCRATCH is unset."""
    scratch = os.environ.get("LASSI_SCRATCH")
    if not scratch:
        raise ValueError("LASSI_SCRATCH is not set; set it or pass --index <file>")
    return Path(scratch) / "rag" / f"{spec.corpus}@{pin['VERSION']}" / INDEX_NAME


def _listing(get: Opener, repo: str, root: list[dict[str, Any]], tops: set[str]) -> dict[str, str]:
    """Return {path: blob sha1} of the regular files at or under each top-level entry of `root` named in `tops`."""
    blobs: dict[str, str] = {}
    for entry in root:
        name = entry["path"]
        if name not in tops:
            continue
        if entry["type"] == "blob" and entry["mode"] in FILE_MODES:
            blobs[name] = entry["sha"]
        elif entry["type"] == "tree":
            for child in _tree(get, repo, entry["sha"], recursive=True):
                if child["type"] == "blob" and child["mode"] in FILE_MODES:
                    blobs[f"{name}/{child['path']}"] = child["sha"]
    return dict(sorted(blobs.items()))


def _tree(get: Opener, repo: str, sha: str, *, recursive: bool) -> list[dict[str, Any]]:
    """Return the entries of one git tree through the GitHub API; refuse a truncated listing."""
    url = API_TREE.format(repo=repo, sha=sha) + ("?recursive=1" if recursive else "")
    document = json.loads(get(url).decode("utf-8"))
    if document.get("truncated"):
        raise ValueError(f"the GitHub listing of {url} is truncated, so the corpus cannot be listed whole")
    return document["tree"]


def _fetch_one(get: Opener, repo: str, commit: str, dest: Path, path: str, blob: str) -> None:
    """Place `path` under `dest` with exactly the bytes of git blob `blob`, downloading it unless already there."""
    target = dest / path
    if target.is_file() and git_blob_sha1(target.read_bytes()) == blob:
        return
    data = get(RAW_FILE.format(repo=repo, commit=commit, path=path))
    if git_blob_sha1(data) != blob:
        raise ValueError(f"{path} at {commit} downloaded with git blob {git_blob_sha1(data)}, not {blob}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)


def _urlopen(url: str) -> bytes:
    """Return the bytes at `url`."""
    with urllib.request.urlopen(url, timeout=TIMEOUT_S) as response:
        return response.read()


def _github_repo(url: str) -> str:
    """Return '<owner>/<name>' of a https://github.com/<owner>/<name>(.git) URL; raise ValueError otherwise."""
    prefix = "https://github.com/"
    if not url.startswith(prefix):
        raise ValueError(f"fetch reads GitHub only, and the pin's URL is {url}")
    repo = url[len(prefix):].removesuffix(".git").strip("/")
    if repo.count("/") != 1:
        raise ValueError(f"{url} names no <owner>/<name> repository")
    return repo


def _literal_root(pattern: str) -> str:
    """Return the directory part of `pattern` before its first glob character ('' for the tree's top)."""
    parts = pattern.split("/")
    literal: list[str] = []
    for part in parts[:-1]:
        if _GLOB & set(part):
            break
        literal.append(part)
    else:
        if not _GLOB & set(parts[-1]):
            return pattern
    return "/".join(literal)


def _refuse_inside_repo(path: Path, what: str) -> None:
    """Raise ValueError when `path` lies in this repository: generated corpus text never enters git."""
    resolved = path.resolve()
    if resolved == REPO or REPO in resolved.parents:
        raise ValueError(f"{what} must lie outside the repository, not at {path}")


def _write(path: Path, data: bytes) -> None:
    """Write `data` to `path` through a sibling temporary file, so a reader never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_bytes(data)
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(prog="build_tt_rag", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    fetch_parser = commands.add_parser("fetch", help="download the spec's files at the pin's commit")
    fetch_parser.add_argument("--dest", required=True, help="directory outside the repository")
    for name in ("build", "check"):
        command = commands.add_parser(name, help=f"{name} the index and manifest")
        command.add_argument("--tree", required=True, help="the pinned tt-metal tree (install or fetch)")
        command.add_argument("--index", help="index file (default $LASSI_SCRATCH/rag/<corpus>@<VERSION>/index.json)")
        command.add_argument("--manifest", help="manifest file (default MANIFEST.yaml beside the corpus spec)")
    for command in commands.choices.values():
        command.add_argument("--corpus", default=str(DEFAULT_CORPUS), help="corpus spec")
    return parser


if __name__ == "__main__":
    sys.exit(main())
