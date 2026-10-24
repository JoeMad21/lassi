"""Build tt-train-v0, SFT records of tt-metal's programming examples at the pin, without Tier A (2026-10-23).

Usage:

    uv run tools/build_tt_train.py fetch --dest <dir>
    uv run tools/build_tt_train.py build --tree <dir> [--out <file>] [--spec <file>] [--manifest <file>]
    uv run tools/build_tt_train.py check --tree <dir> [--out <file>] [--spec <file>] [--manifest <file>]

OQ-076 (1a), the owner's approval of 2026-10-23: train on tt-metal's
example programs at the pin, excluding the five Tier A programs. The spec
(default assets/train/tt-train-v0/dataset.yaml) names the pin, the examples
root, the held-out suite, every example directory under the root with its
task (included) or its reason (excluded), and the prompt template. The tree
is the pinned tt-metal at the pin's COMMIT: an install
($LASSI_TOOLCHAINS/<PREFIX_NAME>, whose lassi-install.txt names the pin) or
a directory `fetch` filled, whose lassi-fetch.txt names the commit and the
data set. Any other tree is refused.

- fetch downloads every regular file under the spec's root at exactly the
  pin's commit, listed through the GitHub git trees API and checked against
  its git blob sha1, into `--dest`, which must lie outside this repository.
- build writes the records (one JSON object {prompt, completion} per line,
  ASCII with LF) to `--out`, by default $LASSI_SCRATCH/train-data/<dataset>.jsonl
  (never in this repository: the records hold tt-metal's text, and no
  tt-metal text is tracked), and the manifest (default manifest.json beside
  the spec), which records the pin, every source file's sha256, the held-out
  check's references and parameters, every exclusion with its reason, and
  the records' sha256, size, and count. lassi train reads the records only
  when their sha256 is the manifest's (lassi.train.data load_corpus).
- check builds in memory and compares with the manifest, and with the
  records file when one is given or found at the default; status 1 names
  each difference, and nothing is written.

Agent Rule 5, the rules build enforces (the spec's comments state them):
1. every eval and unassigned item of the suite has its example directory
   (the directory above kernels/ in each upstream kernel path it lists)
   excluded, and no included directory is, holds, or lies in one;
2. the spec excludes each duplicate of a Tier A task with its reason (a
   judgment the spec records, which no rule computes);
3. no record (prompt or completion) holds a run of run_lines consecutive
   non-blank lines (each stripped, inner whitespace collapsed) of a tracked
   Tier A reference, every run counted, a run of braces too;
4. no completion file is byte-identical to a file of a Tier A example
   directory, or shares at least near_copy of its runs with those files.
A code file (.cpp, .hpp, .h) under the root that no listed directory covers
is refused, so a new example is reviewed before it is trained on; a file
belongs to the deepest listed directory that holds it. The same tree and
spec always give the same records and manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import string
import sys
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from lassi.bench import load_suite  # noqa: E402
from lassi.bench.registry import HELD_SPLITS  # noqa: E402
from lassi.core.files import render_file_blocks  # noqa: E402
from lassi.toolchains import pins  # noqa: E402
from lassi.toolchains.ttmetal_build import check_install_record  # noqa: E402

DEFAULT_SPEC = REPO / "assets" / "train" / "tt-train-v0" / "dataset.yaml"
MANIFEST_NAME = "manifest.json"
GENERATOR = "tools/build_tt_train.py"
FETCH_RECORD = "lassi-fetch.txt"
RECORDS_DIR = "train-data"
FIELDS = ("prompt", "completion")
CODE_SUFFIXES = (".cpp", ".hpp", ".h")
SPEC_KEYS = {"dataset", "pin", "root", "suite", "run_lines", "near_copy", "prompt", "include", "exclude"}
# An included entry's required keys, and the optional one: why an example that resembles a Tier A task stays.
ENTRY_KEYS = {"task", "described_from"}
ENTRY_OPTIONAL = {"judgment"}
API_TREE = "https://api.github.com/repos/{repo}/git/trees/{sha}"
RAW_FILE = "https://raw.githubusercontent.com/{repo}/{commit}/{path}"
FILE_MODES = ("100644", "100755")
TIMEOUT_S = 60
WORKERS = 8
KERNEL_DIR = "/kernels/"

Opener = Callable[[str], bytes]


@dataclass(frozen=True)
class Entry:
    """An included example: its task in this project's words, where the task was read from, and any inclusion judgment.

    `judgment` says why an example that resembles a Tier A task is not a
    duplicate of it (rule 2); None when no Tier A task is near.
    """

    task: str
    described_from: str
    judgment: str | None = None


@dataclass(frozen=True)
class Spec:
    """A checked data set spec (the module docstring); every field is required."""

    dataset: str
    pin: str
    root: str
    suite: str
    run_lines: int
    near_copy: float
    prompt: str
    include: Mapping[str, Entry]
    exclude: Mapping[str, str]


@dataclass(frozen=True)
class HeldOut:
    """The suite the data set is built without: its held items, their example directories, and its references.

    tier_dirs maps each example directory (a tree path) to its item;
    references maps each tracked program's repository path to its text.
    """

    suite: str
    suite_path: str
    suite_sha256: str
    items: tuple[str, ...]
    tier_dirs: Mapping[str, str]
    references: Mapping[str, str]


@dataclass(frozen=True)
class Build:
    """A built data set: the records in example order, each example's record, the exclusions, and the Tier A files."""

    records: tuple[dict[str, str], ...]
    examples: list[dict[str, Any]]
    excluded: list[dict[str, Any]]
    upstream: list[dict[str, str]]


def main(argv: list[str] | None = None) -> int:
    """Run one subcommand; return 0 on success, 1 when check finds a difference, 2 on a refusal."""
    args = _parser().parse_args(argv)
    try:
        spec_path = Path(args.spec).resolve()
        spec = load_spec(spec_path)
        pin = pins.read_pin(spec.pin)
        if args.command == "fetch":
            count = fetch(Path(args.dest), spec, pin)
            print(f"build_tt_train: {count} files of {pin['NAME']} at {pin['COMMIT']} in {args.dest}")
            return 0
        manifest_path = Path(args.manifest) if args.manifest else spec_path.parent / MANIFEST_NAME
        out = _records_path(args, spec)
        if args.command == "build" and out is not None:
            _refuse_inside_repo(out, "the records file")
        document, data = build_tree(Path(args.tree), spec_path, spec, pin)
    except (OSError, ValueError) as error:
        print(f"build_tt_train: {error}", file=sys.stderr)
        return 2
    if args.command == "check" or out is None:
        return check(document, manifest_path, out)
    _write(out, data)
    _write(manifest_path, manifest_text(document).encode("ascii"))
    records = document["records"]
    print(f"build_tt_train: {records['count']} records, {len(document['excluded'])} directories excluded; "
          f"{out} sha256 {records['sha256']} ({records['bytes']} bytes)")
    return 0


def load_spec(path: Path) -> Spec:
    """Read and check a data set spec; ValueError naming the key for any missing, unknown, or malformed one."""
    data = yaml.safe_load(Path(path).read_bytes().decode("ascii"))
    if not isinstance(data, dict) or set(data) != SPEC_KEYS:
        found = sorted(data) if isinstance(data, dict) else data
        raise ValueError(f"{path}: a spec holds exactly {sorted(SPEC_KEYS)}, not {found}")
    run_lines, near_copy = data["run_lines"], data["near_copy"]
    if not isinstance(run_lines, int) or isinstance(run_lines, bool) or run_lines < 2:
        raise ValueError(f"{path}: run_lines must be an integer of at least 2, not {run_lines!r}")
    if not isinstance(near_copy, (int, float)) or isinstance(near_copy, bool) or not 0 < near_copy <= 1:
        raise ValueError(f"{path}: near_copy must be a fraction above 0 and at most 1, not {near_copy!r}")
    include: dict[str, Entry] = {}
    for name, entry in (data["include"] or {}).items():
        if not isinstance(entry, dict) or not ENTRY_KEYS <= set(entry) <= ENTRY_KEYS | ENTRY_OPTIONAL:
            raise ValueError(f"{path}: include.{name} must hold {sorted(ENTRY_KEYS)} and at most "
                             f"{sorted(ENTRY_OPTIONAL)}")
        judgment = entry.get("judgment")
        include[str(name)] = Entry(task=" ".join(str(entry["task"]).split()), described_from=entry["described_from"],
                                   judgment=None if judgment is None else " ".join(str(judgment).split()))
    exclude = {str(name): " ".join(str(reason).split()) for name, reason in (data["exclude"] or {}).items()}
    both = sorted(set(include) & set(exclude))
    if both:
        raise ValueError(f"{path}: {both[0]} is both included and excluded")
    return Spec(dataset=data["dataset"], pin=data["pin"], root=data["root"].strip("/"), suite=data["suite"],
                run_lines=run_lines, near_copy=float(near_copy), prompt=data["prompt"], include=include,
                exclude=exclude)


def load_held_out(spec: Spec, repo: Path = REPO) -> HeldOut:
    """Return the held-out suite the spec names: its eval and unassigned items, their directories, its references."""
    path = repo / spec.suite
    suite = load_suite(path)
    items = tuple(sorted(name for name, item in suite.items.items() if item.split in HELD_SPLITS))
    tier_dirs: dict[str, str] = {}
    references: dict[str, str] = {}
    for name, item in sorted(suite.items.items()):
        for language in item.languages.values():
            if language.tracked:
                for file in language.files:
                    where = f"{language.dir}/{file}"
                    references[where] = (repo / where).read_bytes().decode("utf-8")
            for entry in language.support.values():
                if not entry.tracked:
                    head, marker, _ = entry.path.partition(KERNEL_DIR)
                    if not marker:
                        raise ValueError(f"{spec.suite}: {name}'s upstream path {entry.path} has no kernels/ part")
                    tier_dirs.setdefault(head, name)
    return HeldOut(suite=suite.name, suite_path=spec.suite, suite_sha256=_sha256(path.read_bytes()), items=items,
                   tier_dirs=dict(sorted(tier_dirs.items())), references=dict(sorted(references.items())))


def runs(text: str, size: int) -> list[tuple[str, ...]]:
    """Return every run of `size` consecutive non-blank lines of `text`, each stripped and whitespace-collapsed."""
    lines = [" ".join(line.split()) for line in text.split("\n") if line.strip()]
    return [tuple(lines[index:index + size]) for index in range(len(lines) - size + 1)]


def build(files: Mapping[str, bytes], spec: Spec, held: HeldOut, version: str) -> Build:
    """Return the data set built from the tree's `files` ({tree path: bytes}); ValueError for any rule it breaks."""
    owners = _owners(files, spec)
    tiers = _check_tier_dirs(spec, held)
    upstream = {path: data for path, data in sorted(files.items())
                if _code(path) and any(path.startswith(f"{tier}/") for tier in held.tier_dirs)}
    reference_runs = _run_index(held.references, spec.run_lines)
    upstream_runs = set(_run_index({path: _text(data) for path, data in upstream.items()}, spec.run_lines))
    upstream_bytes = {data: path for path, data in upstream.items()}
    records: list[dict[str, str]] = []
    examples: list[dict[str, Any]] = []
    for number, (name, entry) in enumerate(spec.include.items(), start=1):
        if name in tiers:
            raise ValueError(f"include.{name} is the example directory of the held item {tiers[name]} (Agent Rule 5)")
        owned = {path: files[path] for path in sorted(owners) if owners[path] == name}
        if not owned:
            raise ValueError(f"include.{name}: no code file under {spec.root}/{name}")
        record, meta = _record(name, entry, owned, spec, version)
        _check_runs(name, record, reference_runs, spec.run_lines)
        meta["upstream_overlap"] = _check_copies(name, owned, upstream_bytes, upstream_runs, spec)
        records.append(record)
        examples.append({"example": name, "record": number, **meta})
    excluded = [
        {"example": name, "reason": reason,
         "files": _file_list({path: files[path] for path in sorted(owners) if owners[path] == name})}
        for name, reason in spec.exclude.items()
    ]
    return Build(records=tuple(records), examples=examples, excluded=excluded, upstream=_file_list(upstream, False))


def _owners(files: Mapping[str, bytes], spec: Spec) -> dict[str, str]:
    """Return {tree path: listed directory} for every code file under the root; refuse an unlisted one."""
    listed = sorted({*spec.include, *spec.exclude}, key=len, reverse=True)
    owners: dict[str, str] = {}
    prefix = f"{spec.root}/"
    for path in sorted(files):
        if not path.startswith(prefix) or not _code(path):
            continue
        relative = path[len(prefix):]
        owner = next((name for name in listed if relative.startswith(f"{name}/")), None)
        if owner is None:
            raise ValueError(f"{path} lies in no directory the spec lists; list its example under include or exclude")
        owners[path] = owner
    return owners


def _check_tier_dirs(spec: Spec, held: HeldOut) -> dict[str, str]:
    """Return {directory under the root: held item}; refuse a held item whose directory is not excluded (rule 1)."""
    prefix = f"{spec.root}/"
    tiers = {tier[len(prefix):] if tier.startswith(prefix) else tier: item for tier, item in held.tier_dirs.items()}
    for directory, item in tiers.items():
        if directory not in spec.exclude:
            raise ValueError(f"the held item {item} of suite {held.suite} has its example directory {directory}, "
                             "which the spec does not exclude (Agent Rule 5)")
        for name in spec.include:
            if name.startswith(f"{directory}/") or directory.startswith(f"{name}/"):
                raise ValueError(f"include.{name} holds or lies in {directory}, the example directory of the held "
                                 f"item {item} (Agent Rule 5)")
    unplaced = sorted(set(held.items) - set(tiers.values()))
    if unplaced:
        raise ValueError(f"the held item {unplaced[0]} of suite {held.suite} names no example directory to exclude")
    return tiers


def _record(name: str, entry: Entry, owned: Mapping[str, bytes], spec: Spec,
            version: str) -> tuple[dict[str, str], dict[str, Any]]:
    """Return the example's record and its manifest entry (without the overlap figure)."""
    base = f"{spec.root}/{name}/"
    texts = {path[len(base):]: _text(data) for path, data in owned.items()}
    completion = render_file_blocks(texts)
    paths = ", ".join(sorted(texts))
    prompt = string.Template(spec.prompt).substitute(version=version, task=entry.task, files=paths)
    meta = {"described_from": entry.described_from, "judgment": entry.judgment, "prompt_chars": len(prompt),
            "completion_chars": len(completion), "files": _file_list(owned)}
    return {"prompt": prompt, "completion": completion}, meta


def _check_runs(name: str, record: Mapping[str, str], index: Mapping[tuple[str, ...], str], size: int) -> None:
    """Refuse a record whose prompt or completion holds a run of a tracked Tier A reference (rule 3)."""
    for field in FIELDS:
        for run in runs(record[field], size):
            if run in index:
                raise ValueError(f"include.{name}: its {field} holds the run {' | '.join(run)!r} of {index[run]}; "
                                 "exclude the example (Agent Rule 5)")


def _check_copies(name: str, owned: Mapping[str, bytes], upstream_bytes: Mapping[bytes, str],
                  upstream_runs: set[tuple[str, ...]], spec: Spec) -> float:
    """Refuse a completion file that copies a Tier A file (rule 4); return the largest overlap fraction found."""
    largest = 0.0
    for path, data in owned.items():
        if data in upstream_bytes:
            raise ValueError(f"include.{name}: {path} is byte-identical to {upstream_bytes[data]}, a file of a Tier A "
                             "example directory (Agent Rule 5)")
        found = runs(_text(data), spec.run_lines)
        fraction = sum(run in upstream_runs for run in found) / len(found) if found else 0.0
        if fraction >= spec.near_copy:
            raise ValueError(f"include.{name}: {path} is a near copy of the Tier A example files: {fraction:.2f} of "
                             f"its runs are theirs, at least near_copy {spec.near_copy} (Agent Rule 5)")
        largest = max(largest, fraction)
    return round(largest, 4)


def _run_index(texts: Mapping[str, str], size: int) -> dict[tuple[str, ...], str]:
    """Return every run of every text with the first text (in path order) that holds it."""
    index: dict[tuple[str, ...], str] = {}
    for path in sorted(texts):
        for run in runs(texts[path], size):
            index.setdefault(run, path)
    return index


def records_bytes(records: Sequence[Mapping[str, str]]) -> bytes:
    """Return the records as JSON Lines: one object per line, fields in order, ASCII escapes, each line ending in LF."""
    lines = [json.dumps({field: record[field] for field in FIELDS}, ensure_ascii=True) + "\n" for record in records]
    return "".join(lines).encode("ascii")


def build_tree(tree: Path, spec_path: Path, spec: Spec, pin: Mapping[str, str]) -> tuple[dict[str, Any], bytes]:
    """Return (manifest document, records bytes) for `tree`; refuse a tree that is not the pinned one."""
    check_tree(tree, pin, spec.dataset)
    held = load_held_out(spec)
    suite = load_suite(REPO / spec.suite)
    if suite.commit != pin["COMMIT"]:
        raise ValueError(f"{spec.suite} pins {suite.commit}, not {spec.pin}'s COMMIT {pin['COMMIT']}")
    built = build(tree_files(tree, spec.root), spec, held, pin["VERSION"])
    data = records_bytes(built.records)
    document = manifest_document(built, spec, held, pin, _sha256(spec_path.read_bytes()), data, spec_path)
    return document, data


def manifest_document(built: Build, spec: Spec, held: HeldOut, pin: Mapping[str, str], spec_sha256: str,
                      data: bytes, spec_path: Path) -> dict[str, Any]:
    """Return the manifest: the pin, the records' digest and size, the held-out check, and every example's files."""
    try:
        spec_name = spec_path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        spec_name = spec_path.name
    return {
        "corpus": spec.dataset,
        "generator": GENERATOR,
        "note": ("Generated from the spec and the pinned tree; do not edit. The records it pins are generated under "
                 "$LASSI_SCRATCH/train-data, never in the repository, since they hold tt-metal's text."),
        "spec": spec_name,
        "spec_sha256": spec_sha256,
        "source": {"pin": spec.pin, "repo": pin["URL"], "commit": pin["COMMIT"], "version": pin["VERSION"],
                   "root": spec.root},
        "records": {"file": f"{spec.dataset}.jsonl", "sha256": _sha256(data), "bytes": len(data),
                    "count": len(built.records), "fields": list(FIELDS),
                    "prompt_chars": sum(len(record["prompt"]) for record in built.records),
                    "completion_chars": sum(len(record["completion"]) for record in built.records)},
        "held_out": {"suite": held.suite, "suite_manifest": held.suite_path, "suite_sha256": held.suite_sha256,
                     "items": list(held.items), "run_lines": spec.run_lines, "near_copy": spec.near_copy,
                     "references": [{"path": path, "sha256": _sha256(text.encode("utf-8"))}
                                    for path, text in held.references.items()],
                     "example_dirs": [{"path": path, "item": item} for path, item in held.tier_dirs.items()],
                     "upstream": built.upstream},
        "examples": built.examples,
        "excluded": built.excluded,
    }


def manifest_text(document: Mapping[str, Any]) -> str:
    """Return the manifest as JSON: two-space indent, ASCII, keys in document order, a final LF."""
    return json.dumps(dict(document), indent=2, ensure_ascii=True) + "\n"


def check(document: Mapping[str, Any], manifest_path: Path, out: Path | None) -> int:
    """Return 0 when the manifest and any records file equal this build; else name each difference."""
    problems: list[str] = []
    try:
        recorded = json.loads(manifest_path.read_bytes().decode("ascii"))
    except (OSError, ValueError) as error:
        recorded, problems = {}, [f"{manifest_path}: {error}"]
    for key in document:
        if recorded and recorded.get(key) != json.loads(json.dumps(document[key])):
            problems.append(f"{manifest_path}: {key} differs from this build")
    if out is not None and out.is_file():
        digest = _sha256(out.read_bytes())
        if digest != document["records"]["sha256"]:
            problems.append(f"{out} has sha256 {digest}, this build {document['records']['sha256']}")
    else:
        print(f"build_tt_train: no records at {out}; checked the manifest only")
    for problem in problems:
        print(f"build_tt_train: {problem}", file=sys.stderr)
    if not problems:
        print(f"build_tt_train: {manifest_path} matches this build (records sha256 {document['records']['sha256']})")
    return 1 if problems else 0


def check_tree(tree: Path, pin: Mapping[str, str], dataset: str) -> None:
    """Raise ValueError unless `tree` is an install of the pin or a fetch of the pin's commit for `dataset`."""
    record = tree / FETCH_RECORD
    if record.is_file():
        lines = record.read_bytes().decode("utf-8", errors="replace").splitlines()
        if lines[:1] == [pin["COMMIT"]] and len(lines) > 1 and lines[1].startswith(f"dataset {dataset}:"):
            return
        raise ValueError(f"{record} does not name commit {pin['COMMIT']} and data set {dataset}; fetch the tree again "
                         "with uv run tools/build_tt_train.py fetch")
    try:
        check_install_record(tree, pin)
    except ValueError as error:
        raise ValueError(f"{error}; or fetch the files with uv run tools/build_tt_train.py fetch") from None


def tree_files(tree: Path, root: str) -> dict[str, bytes]:
    """Return {tree path: bytes} of every regular file under `root` in `tree`, never through a symbolic link."""
    base = tree / root
    if not base.is_dir() or any((tree / part).is_symlink() for part in _prefixes(root)):
        raise ValueError(f"{base} is not a directory of the tree")
    found: dict[str, bytes] = {}
    for directory, _, names in os.walk(base, followlinks=False):
        for name in names:
            path = Path(directory) / name
            if not path.is_symlink() and path.is_file():
                found[path.relative_to(tree).as_posix()] = path.read_bytes()
    return dict(sorted(found.items()))


def fetch(dest: Path, spec: Spec, pin: Mapping[str, str], opener: Opener | None = None) -> int:
    """Fill `dest` with every regular file under the spec's root at the pin's commit; return how many it holds.

    Each file is checked against the blob sha1 of the commit's git tree, so
    the bytes are the commit's exactly. `opener` (default: urllib) returns a
    URL's bytes; tests pass their own.
    """
    _refuse_inside_repo(dest, "the fetch destination")
    dest.mkdir(parents=True, exist_ok=True)
    get = opener or _urlopen
    repo, commit = _github_repo(pin["URL"]), pin["COMMIT"]
    blobs = _listing(get, repo, commit, spec.root)
    with ThreadPoolExecutor(WORKERS) as pool:
        list(pool.map(lambda item: _fetch_one(get, repo, commit, dest, *item), sorted(blobs.items())))
    text = (f"{commit}\ndataset {spec.dataset}: files of {pin['URL']} at {commit} under {spec.root}, each checked "
            "against its git blob\n")
    (dest / FETCH_RECORD).write_bytes(text.encode("ascii"))
    return len(blobs)


def _listing(get: Opener, repo: str, commit: str, root: str) -> dict[str, str]:
    """Return {tree path: blob sha1} of the regular files under `root`, walking the git trees down to it."""
    sha = commit
    for part in root.split("/"):
        entries = _tree(get, repo, sha, recursive=False)
        match = next((entry for entry in entries if entry["path"] == part and entry["type"] == "tree"), None)
        if match is None:
            raise ValueError(f"{root} is not a directory of {repo} at {commit}")
        sha = match["sha"]
    return {f"{root}/{entry['path']}": entry["sha"] for entry in _tree(get, repo, sha, recursive=True)
            if entry["type"] == "blob" and entry["mode"] in FILE_MODES}


def _tree(get: Opener, repo: str, sha: str, *, recursive: bool) -> list[dict[str, Any]]:
    """Return the entries of one git tree through the GitHub API; refuse a truncated listing."""
    url = API_TREE.format(repo=repo, sha=sha) + ("?recursive=1" if recursive else "")
    document = json.loads(get(url).decode("utf-8"))
    if document.get("truncated"):
        raise ValueError(f"the GitHub listing of {url} is truncated, so the examples cannot be listed whole")
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


def git_blob_sha1(data: bytes) -> str:
    """Return the git blob id of `data`: the sha1 of b'blob <size>\\0' followed by the bytes."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _records_path(args: argparse.Namespace, spec: Spec) -> Path | None:
    """Return --out, else $LASSI_SCRATCH/train-data/<dataset>.jsonl; None for check without either."""
    if args.out:
        return Path(args.out)
    scratch = os.environ.get("LASSI_SCRATCH", "")
    if scratch:
        return Path(scratch) / RECORDS_DIR / f"{spec.dataset}.jsonl"
    if args.command == "check":
        return None
    raise ValueError("LASSI_SCRATCH is not set; set it or pass --out <file> outside the repository")


def _file_list(files: Mapping[str, bytes], sizes: bool = True) -> list[dict[str, Any]]:
    """Return [{path, sha256[, bytes]}] for `files` in path order."""
    return [{"path": path, "sha256": _sha256(data), **({"bytes": len(data)} if sizes else {})}
            for path, data in sorted(files.items())]


def _code(path: str) -> bool:
    """Return True for a code file: a .cpp, .hpp, or .h file."""
    return path.endswith(CODE_SUFFIXES)


def _text(data: bytes) -> str:
    """Return a source file's text, decoded as UTF-8 (strict)."""
    return data.decode("utf-8")


def _sha256(data: bytes) -> str:
    """Return the sha256 hex digest of `data`."""
    return hashlib.sha256(data).hexdigest()


def _prefixes(relative: str) -> list[str]:
    """Return every leading part of a POSIX path, itself included: a/b/c gives a, a/b, a/b/c."""
    parts = [part for part in relative.split("/") if part]
    return ["/".join(parts[:count]) for count in range(1, len(parts) + 1)]


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


def _refuse_inside_repo(path: Path, what: str) -> None:
    """Raise ValueError when `path` lies in this repository: generated text from tt-metal never enters git."""
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
    parser = argparse.ArgumentParser(prog="build_tt_train", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    fetch_parser = commands.add_parser("fetch", help="download the examples at the pin's commit")
    fetch_parser.add_argument("--dest", required=True, help="directory outside the repository")
    for name in ("build", "check"):
        command = commands.add_parser(name, help=f"{name} the records and the manifest")
        command.add_argument("--tree", required=True, help="the pinned tt-metal tree (install or fetch)")
        command.add_argument("--out", help="records file (default $LASSI_SCRATCH/train-data/<dataset>.jsonl)")
        command.add_argument("--manifest", help="manifest file (default manifest.json beside the spec)")
    for command in commands.choices.values():
        command.add_argument("--spec", default=str(DEFAULT_SPEC), help="data set spec")
    return parser


if __name__ == "__main__":
    sys.exit(main())
