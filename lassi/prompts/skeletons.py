"""Host program skeletons: a generic program a prompt carries for the model to fill (OQ-096, option (a)).

A skeleton is a directory `<assets root>/SKELETONS_DIR/<name>/` holding
MANIFEST.yaml, HOST_FILE (a complete program in the skeleton's language with
every API call in the form its pin declares, and blocks between the lines
`// BLANK <name>: <what to fill>` and `// END BLANK <name>` for what differs
per program), and BLOCK_FILE (the prompt block that carries it, a
string.Template whose one field, BLOCK_FIELD, is the host program's text).
Both files are this repository's own text, written by hand, so both are
tracked, and the manifest beside them pins their bytes.

The manifest holds MANIFEST_KEYS: `skeleton`, the directory's name;
`language`, the language id of the programs it is for (the target language
of every direction a recipe that carries it runs); `pin` and `commit`, the
toolchain pin and commit its calls are written for; `files`, exactly
HOST_FILE and BLOCK_FILE, each with its path, sha256, and size in bytes; and
`sha256`, the skeleton's digest (files_sha256): the sha256 of the lines
"<sha256>  <path>\\n" of the files in the manifest's order, so a change to
either file changes it. check_skeleton names every difference between a
skeleton's files and its manifest; load_skeleton returns the skeleton only
when there is none and its block renders. A change to either file is a
change to the manifest too: each file's sha256 and bytes, and `sha256`.

Kernel skeletons (task P17.39). After HOST_FILE and BLOCK_FILE, `files`
may list kernel skeletons, each a relative .cpp path with a directory named
KERNEL_DIR in it and no "." or ".." segment (the paths the host skeleton's
CreateKernel calls name), in the order a prompt shows them; the sha256
covers them like the other two. Skeleton.kernels holds them as (path,
text), and the block then uses a second field, KERNELS_FIELD, which renders
them as FILE blocks in that order (lassi.prompts.examples file_blocks). A
skeleton with kernels whose block holds no $kernels, and one without
kernels whose block uses it, are refused. A skeleton without kernels reads
and renders exactly as before.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any

import yaml

from lassi.prompts import assets as prompt_assets
from lassi.prompts.examples import file_blocks

MANIFEST = "MANIFEST.yaml"
# The directory under the assets root that holds the skeletons, and the two files each holds first.
SKELETONS_DIR = "skeletons"
HOST_FILE = "host.cpp"
BLOCK_FILE = "block.txt"
FILES = (HOST_FILE, BLOCK_FILE)
# The field of BLOCK_FILE every skeleton fills: the host program's text.
BLOCK_FIELD = "skeleton"
# The second field, for a skeleton that lists kernel skeletons (task P17.39): them as FILE blocks. A kernel
# skeleton's path holds a directory of this name and ends in this suffix, as a kernel source does
# (lassi.toolchains.ttmetal_build is_kernel_source).
KERNELS_FIELD = "kernels"
KERNEL_DIR = "kernels"
KERNEL_SUFFIX = ".cpp"
MANIFEST_KEYS = ("skeleton", "language", "pin", "commit", "files", "sha256")
_FILE_KEYS = {"path", "sha256", "bytes"}
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_FIX = (f"keep each listed file's sha256 and bytes and the manifest's sha256 (that of the lines "
        f"'<sha256>  <path>' of {HOST_FILE}, {BLOCK_FILE}, and any kernel skeletons) true to the files")


@dataclass(frozen=True)
class Skeleton:
    """One skeleton read from its directory: name, language, pin, commit, host text, block, sha256, and kernels.

    `kernels` holds the kernel skeletons as (path, text) in the manifest's
    order, empty for a host-only skeleton.
    """

    name: str
    language: str
    pin: str
    commit: str
    host: str
    block: str
    sha256: str
    kernels: tuple[tuple[str, str], ...] = ()


def skeleton_dir(name: str, root: Path | None = None) -> Path:
    """Return the directory of skeleton `name`: <root>/SKELETONS_DIR/<name>, root lassi.prompts.assets.default_root().

    The root is read at each call, so a caller that points default_root
    elsewhere is followed. Raises ValueError for a name that is not one
    plain path segment.
    """
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise ValueError(f"a skeleton name must match {_NAME.pattern}, got {name!r}")
    base = prompt_assets.default_root() if root is None else Path(root)
    return base / SKELETONS_DIR / name


def files_sha256(entries: Sequence[tuple[str, str]]) -> str:
    """Return the skeleton's sha256 from its (path, sha256) entries in order: that of "<sha256>  <path>\\n" lines."""
    lines = "".join(f"{digest}  {path}\n" for path, digest in entries)
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def read_manifest(path: Path) -> dict[str, Any]:
    """Return the manifest at `path`; raise ValueError naming it when it is missing or malformed.

    It must hold MANIFEST_KEYS, a plain name in each of skeleton, language,
    and pin, a 40-character lowercase hex commit, a 64-character sha256, and
    `files` listing FILES in that order and then any kernel skeletons
    (is_kernel_path), each with its path, a sha256, and a size in bytes.
    """
    try:
        document = yaml.safe_load(Path(path).read_bytes().decode("utf-8"))
    except FileNotFoundError:
        raise ValueError(f"{path} does not exist; a skeleton directory holds {MANIFEST}, {', '.join(FILES)}") from None
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise ValueError(f"cannot read {path} ({error})") from error
    if not isinstance(document, dict) or not set(MANIFEST_KEYS) <= set(document):
        raise ValueError(f"{path} must hold {', '.join(MANIFEST_KEYS)}")
    for key in ("skeleton", "language", "pin"):
        if not isinstance(document[key], str) or not _NAME.fullmatch(document[key]):
            raise ValueError(f"{path}: {key} must be a plain name, got {document[key]!r}")
    if not isinstance(document["commit"], str) or not re.fullmatch(r"[0-9a-f]{40}", document["commit"]):
        raise ValueError(f"{path}: commit must be a full commit id, got {document['commit']!r}")
    if not isinstance(document["sha256"], str) or not _SHA256.fullmatch(document["sha256"]):
        raise ValueError(f"{path}: sha256 must be a sha256 hex digest, got {document['sha256']!r}")
    files = document["files"]
    if not isinstance(files, list) or [_entry_path(entry) for entry in files[:len(FILES)]] != list(FILES):
        raise ValueError(f"{path}: files must list {', '.join(FILES)} first, in that order")
    for entry in files[len(FILES):]:
        if not is_kernel_path(_entry_path(entry)):
            raise ValueError(f"{path}: after {', '.join(FILES)}, files lists only kernel skeletons, each a relative "
                             f"{KERNEL_SUFFIX} path with a directory named {KERNEL_DIR}, got {_entry_path(entry)!r}")
    for entry in files:
        if set(entry) != _FILE_KEYS or not isinstance(entry["sha256"], str) or not _is_size(entry["bytes"]):
            raise ValueError(f"{path}: a file entry must hold {sorted(_FILE_KEYS)}: {entry!r}")
    return document


def is_kernel_path(path: Any) -> bool:
    """Return True for a kernel skeleton's path: relative POSIX, a KERNEL_SUFFIX file under a KERNEL_DIR directory.

    No segment may be empty, ".", or "..", and no backslash may appear.
    """
    if not isinstance(path, str) or not path or path.startswith("/") or "\\" in path:
        return False
    parts = path.split("/")
    if any(part in ("", ".", "..") for part in parts):
        return False
    return parts[-1].endswith(KERNEL_SUFFIX) and KERNEL_DIR in parts[:-1]


def _entry_path(entry: Any) -> Any:
    """Return a file entry's path, or None for an entry that is not a mapping."""
    return entry.get("path") if isinstance(entry, dict) else None


def _is_size(value: Any) -> bool:
    """Return True for an int that is not a bool and is at least 0."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def check_skeleton(tree: Path) -> list[str]:
    """Return one message per difference between the skeleton at `tree` and its MANIFEST.yaml; [] when it matches.

    A listed file that is missing, of another size, or of another sha256 is
    named; so is a manifest whose `sha256` is not files_sha256 of its
    entries, or whose `skeleton` is not the directory's name. A manifest
    that does not read gives its one message.
    """
    tree = Path(tree)
    try:
        document = read_manifest(tree / MANIFEST)
    except ValueError as error:
        return [str(error)]
    problems: list[str] = []
    if document["skeleton"] != tree.name:
        problems.append(f"{tree / MANIFEST} names the skeleton {document['skeleton']!r}, not {tree.name!r}")
    for entry in document["files"]:
        problem = _file_problem(tree, entry)
        if problem:
            problems.append(problem)
    digest = files_sha256([(entry["path"], entry["sha256"]) for entry in document["files"]])
    if digest != document["sha256"]:
        problems.append(f"{tree / MANIFEST}: its sha256 {document['sha256']} is not that of its entries, {digest}")
    return problems


def _file_problem(tree: Path, entry: Mapping[str, Any]) -> str:
    """Return what is wrong with one listed file of the skeleton ("" when it holds the manifest's bytes)."""
    path = tree / entry["path"]
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return f"{path} is missing"
    except OSError as error:
        return f"cannot read {path} ({error})"
    if len(data) != entry["bytes"]:
        return f"{path} holds {len(data)} bytes, not the manifest's {entry['bytes']}"
    if hashlib.sha256(data).hexdigest() != entry["sha256"]:
        return f"{path} differs from its sha256 in the manifest"
    return ""


def load_skeleton(tree: Path) -> Skeleton:
    """Return the skeleton at `tree`; raise ValueError naming every problem, so a skeleton off its manifest is unread.

    The files must match the manifest (check_skeleton) and be UTF-8 text,
    and BLOCK_FILE must render with its fields (block_text).
    """
    tree = Path(tree)
    problems = check_skeleton(tree)
    if problems:
        raise ValueError(f"{'; '.join(problems)}; {_FIX}")
    document = read_manifest(tree / MANIFEST)
    kernel_paths = [entry["path"] for entry in document["files"][len(FILES):]]
    try:
        host, block = ((tree / name).read_bytes().decode("utf-8") for name in FILES)
        kernels = tuple((path, (tree / path).read_bytes().decode("utf-8")) for path in kernel_paths)
    except UnicodeDecodeError as error:
        raise ValueError(f"{tree}: a skeleton file is not UTF-8 text ({error})") from None
    skeleton = Skeleton(
        name=document["skeleton"],
        language=document["language"],
        pin=document["pin"],
        commit=document["commit"],
        host=host,
        block=block,
        sha256=document["sha256"],
        kernels=kernels,
    )
    block_text(skeleton, where=str(tree / BLOCK_FILE))
    return skeleton


def block_text(skeleton: Skeleton, *, where: str | None = None) -> str:
    """Return the skeleton's block: BLOCK_FILE with $skeleton (the host text) and any $kernels filled, as is.

    $kernels is the kernel skeletons as FILE blocks in their order
    (file_blocks). Raises ValueError naming `where` (default: the
    skeleton's BLOCK_FILE) when the block uses no BLOCK_FIELD, a skeleton
    with kernels has a block without KERNELS_FIELD or one without kernels a
    block with it, the block uses any other placeholder, or a `$` that is
    not a placeholder.
    """
    where = where or f"{skeleton.name}/{BLOCK_FILE}"
    fields = {BLOCK_FIELD: skeleton.host}
    if skeleton.kernels:
        fields[KERNELS_FIELD] = file_blocks(skeleton.kernels)
    template = Template(skeleton.block)
    try:
        named = {match.group("named") or match.group("braced") for match in template.pattern.finditer(skeleton.block)
                 if match.group("named") or match.group("braced")}
        text = template.substitute(fields)
    except KeyError as error:
        if error.args[0] == KERNELS_FIELD:
            raise ValueError(f"{where}: the block uses ${KERNELS_FIELD}, but the skeleton lists no kernel "
                             "skeleton") from None
        allowed = ", ".join(f"${name}" for name in fields)
        raise ValueError(f"{where}: the placeholder ${error.args[0]} is not one of its fields, {allowed}") from None
    except ValueError as error:
        raise ValueError(f"{where}: {error}") from None
    if BLOCK_FIELD not in named:
        raise ValueError(f"{where}: the block holds no ${BLOCK_FIELD}, so it would carry no host program")
    if skeleton.kernels and KERNELS_FIELD not in named:
        raise ValueError(f"{where}: the block holds no ${KERNELS_FIELD}, so it would carry none of the skeleton's "
                         f"{len(skeleton.kernels)} kernel skeletons")
    return text
