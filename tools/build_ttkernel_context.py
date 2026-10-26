"""Build the ttkernel dialect pack and the ttkernel worked example from tt-mlir at the pin (task P17.36).

Usage:

    uv run tools/build_ttkernel_context.py fetch --dest <dir>
    uv run tools/build_ttkernel_context.py build --tree <dir> [--out <assets dir>] [--limit-bytes N]
    uv run tools/build_ttkernel_context.py check --tree <dir> [--out <assets dir>] [--limit-bytes N]

The tree is tt-mlir at toolchains/tt-mlir.pin's COMMIT: an install's source
tree, $LASSI_TOOLCHAINS/<PREFIX_NAME>/src/tt-mlir, or a directory `fetch`
filled, whose lassi-fetch.txt names the commit. `fetch` downloads exactly
the SOURCES and EXAMPLE_FILES at the commit from GitHub, each checked
against the git blob sha1 that the git trees API lists for it at the
commit, into `--dest` (outside this repository).

`build` writes two trees under `--out` (default: assets/):

- context/ttkernel-5f396fd6/: the context pack PACK (lassi.prompts.assets),
  two entries whose keys end in the language ttkernel: ttkernel_ops, one
  block per op of TTKernelOps.td (its mnemonic, def name, summary,
  operands, results, and assembly format, with descriptions left out), the
  ops of CORE_OPS first and then the rest in file order, each block kept
  whole until the pack would pass --limit-bytes (LIMIT_BYTES); and
  ttkernel_types, TTKernelOpsTypes.td and TTKernelOpsEnums.td with their
  license headers left out. cut_by_limit names, in fill order, every op
  skipped as it did not fit.
- examples/ttkernel-examples-5f396fd6/: the worked examples pack EXAMPLES
  (lassi.prompts.examples), one example, ttkernel_translate, holding
  tt-mlir's lit tests that lower ttkernel to C++ at the pin
  (test/ttmlir/Translate/TTKernel/ttkernel_noc.mlir and
  ttkernel_compute.mlir), copied byte for byte.

Only the two MANIFEST.yaml files are tracked; the fragments are tt-mlir's
text and are gitignored, as OQ-018 keeps upstream text out of the
repository. Neither tree holds anything of a tt-pairs-v0 item: the sources
are tt-mlir's dialect definitions and its own tests (Agent Rule 5). `check`
builds in memory and names every difference with the files under --out;
status 1 on a difference, 2 on a refusal. The same tree always gives the
same bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from lassi.prompts import examples as examples_pack  # noqa: E402
from lassi.toolchains import pins  # noqa: E402

GENERATOR = "tools/build_ttkernel_context.py"
PIN = "tt-mlir"
PACK = "ttkernel-5f396fd6"
EXAMPLES = "ttkernel-examples-5f396fd6"
LANGUAGE = "ttkernel"
LIMIT_BYTES = 40960  # the tt-metalium pack's limit, which summarize_context has carried (sweep6's api recipes)
FETCH_RECORD = "lassi-fetch.txt"
IR = "include/ttmlir/Dialect/TTKernel/IR"
OPS_SOURCE = f"{IR}/TTKernelOps.td"
TYPE_SOURCES = (f"{IR}/TTKernelOpsTypes.td", f"{IR}/TTKernelOpsEnums.td")
SOURCES = (OPS_SOURCE, *TYPE_SOURCES)
EXAMPLE_NAME = "ttkernel_translate"
EXAMPLE_FILES = (
    "test/ttmlir/Translate/TTKernel/ttkernel_noc.mlir",
    "test/ttmlir/Translate/TTKernel/ttkernel_compute.mlir",
)
# The ops a data-movement or an eltwise or matmul compute kernel uses, carried first (by mnemonic).
CORE_OPS = (
    "get_arg_val", "get_common_arg_val", "get_compile_time_arg_val", "TensorAccessorArgs", "TensorAccessor",
    "tensor_accessor.get_noc_addr", "get_noc_addr", "noc_async_read_tile", "noc_async_write_tile", "noc_async_read",
    "noc_async_write", "noc_async_read_barrier", "noc_async_write_barrier", "get_tile_size", "cb_reserve_back",
    "get_write_ptr", "cb_push_back", "cb_wait_front", "get_read_ptr", "cb_pop_front", "compute_kernel_hw_startup",
    "binary_op_init_common", "init_sfpu", "tile_regs_acquire", "tile_regs_commit", "tile_regs_wait",
    "tile_regs_release", "copy_tile_init", "copy_tile", "add_tiles_init", "add_tiles", "exp_tile_init", "exp_tile",
    "mm_init", "matmul_tiles", "pack_tile",
)
_DEF = re.compile(r'^def (TTKernel_\w+Op)\s*:\s*TTKernel_\w*Op<\s*"([^"]+)"', re.MULTILINE)
_LET = re.compile(r"^\s*let (summary|arguments|results|assemblyFormat)\s*=\s*", re.MULTILINE)
_LICENSE = re.compile(r"\A(?://[^\n]*\n)+\s*")
OPS_HEAD = (
    "// The ops of the ttkernel dialect of tt-mlir {version}, commit {commit},\n"
    "// from {source}: for each op its textual name, its TableGen def, its summary, its operands (ins), its\n"
    "// results (outs), and its assembly format, in which $name is an operand or attribute and\n"
    "// functional-type(operands, results) is the trailing `: (operand types) -> result types`. Descriptions\n"
    "// are left out, and ops past the pack's size limit are left out too (the manifest's cut_by_limit).\n"
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one subcommand; 0 on success, 1 when check finds a difference, 2 on a refusal."""
    args = _parser().parse_args(argv)
    try:
        pin = pins.read_pin(PIN)
        if args.command == "fetch":
            fetch(Path(args.dest), pin)
            return 0
        built = build(Path(args.tree), pin, args.limit_bytes)
        out = Path(args.out)
        if args.command == "build":
            write(built, out)
            return 0
        problems = compare(built, out)
    except (OSError, ValueError) as error:
        print(f"build_ttkernel_context: {error}", file=sys.stderr)
        return 2
    for problem in problems:
        print(problem)
    return 1 if problems else 0


def _parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    fetcher = sub.add_parser("fetch")
    fetcher.add_argument("--dest", required=True)
    for name in ("build", "check"):
        command = sub.add_parser(name)
        command.add_argument("--tree", required=True)
        command.add_argument("--out", default=str(REPO / "assets"))
        command.add_argument("--limit-bytes", type=int, default=LIMIT_BYTES)
    return parser


def blob_sha1(data: bytes) -> str:
    """Return git's blob sha1 of `data`."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _get(url: str) -> bytes:
    """Return the body of an https GET."""
    request = urllib.request.Request(url, headers={"User-Agent": "lassi-build-ttkernel-context"})
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 (a fixed https URL)
        return bytes(response.read())


def fetch(dest: Path, pin: Mapping[str, str]) -> None:
    """Download SOURCES and EXAMPLE_FILES at the pin's COMMIT into `dest`, each checked by its blob sha1."""
    if dest.resolve() == REPO or REPO in dest.resolve().parents:
        raise ValueError(f"fetch writes tt-mlir's text outside this repository, not under {REPO}")
    commit, repo = pin["COMMIT"], pin["URL"].removesuffix(".git").removeprefix("https://github.com/")
    listing = json.loads(_get(f"https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1"))
    blobs = {entry["path"]: entry["sha"] for entry in listing["tree"] if entry["type"] == "blob"}
    for path in (*SOURCES, *EXAMPLE_FILES):
        if path not in blobs:
            raise ValueError(f"{path} is not in tt-mlir's tree at {commit}")
        data = _get(f"https://raw.githubusercontent.com/{repo}/{commit}/{path}")
        if blob_sha1(data) != blobs[path]:
            raise ValueError(f"{path} at {commit} does not match its git blob sha1 {blobs[path]}")
        (dest / path).parent.mkdir(parents=True, exist_ok=True)
        (dest / path).write_bytes(data)
    (dest / FETCH_RECORD).write_bytes(f"tt-mlir {commit} {GENERATOR}\n".encode("ascii"))


def check_tree(tree: Path, pin: Mapping[str, str]) -> None:
    """Refuse a tree that is not tt-mlir at the pin: a fetched one's record or an install's must name COMMIT."""
    records = [tree / FETCH_RECORD, tree.parent.parent / "lassi-install.txt"]
    for record in records:
        if record.is_file():
            words = record.read_text(encoding="utf-8").split()[:2]
            if words == ["tt-mlir", pin["COMMIT"]]:
                return
            raise ValueError(f"{record} names {' '.join(words)!r}, not tt-mlir {pin['COMMIT']}")
    raise ValueError(f"{tree} holds no {FETCH_RECORD} and is not an install's src/tt-mlir; fetch it first")


def op_blocks(text: str) -> list[tuple[str, str]]:
    """Return (mnemonic, block) for every op def of TTKernelOps.td, in file order (the module docstring)."""
    starts = [match for match in _DEF.finditer(text)]
    blocks: list[tuple[str, str]] = []
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        body = text[match.start():end]
        lines = [f"ttkernel.{match.group(2)}  ({match.group(1)})"]
        for field in ("summary", "arguments", "results", "assemblyFormat"):
            value = _let_value(body, field)
            if value:
                lines.append(f"  {field}: {value}")
        blocks.append((match.group(2), "\n".join(lines) + "\n"))
    return blocks


def _let_value(body: str, field: str) -> str:
    """Return one `let <field> = ...;` value of a def body on one line, whitespace collapsed; "" when absent."""
    for match in _LET.finditer(body):
        if match.group(1) != field:
            continue
        rest = body[match.end():]
        if rest.startswith("[{"):
            value = rest[2:rest.find("}]")]
        else:
            value = rest[: _statement_end(rest)]
        return " ".join(value.split()).strip('"')
    return ""


def _statement_end(text: str) -> int:
    """Return the index of the `;` that ends a statement, outside parentheses and quotes."""
    depth, quoted = 0, False
    for index, char in enumerate(text):
        if char == '"':
            quoted = not quoted
        elif not quoted and char == "(":
            depth += 1
        elif not quoted and char == ")":
            depth -= 1
        elif not quoted and char == ";" and depth <= 0:
            return index
    return len(text)


def build(tree: Path, pin: Mapping[str, str], limit: int) -> dict[str, dict[str, bytes]]:
    """Return {tree name: {file path: bytes}} for both trees, manifests included."""
    check_tree(tree, pin)
    if limit < 1:
        raise ValueError(f"--limit-bytes must be at least 1, got {limit}")
    read = {path: (tree / path).read_bytes() for path in (*SOURCES, *EXAMPLE_FILES)}
    types = "\n".join(_LICENSE.sub("", read[path].decode("utf-8")) for path in TYPE_SOURCES)
    types_text = f"// The types and enums of the ttkernel dialect at {pin['COMMIT']}.\n{types}"
    head = OPS_HEAD.format(version=pin["VERSION"], commit=pin["COMMIT"], source=OPS_SOURCE)
    blocks = op_blocks(read[OPS_SOURCE].decode("utf-8"))
    order = sorted(blocks, key=lambda item: (CORE_OPS.index(item[0]) if item[0] in CORE_OPS else len(CORE_OPS)))
    kept, cut, used = [], [], len(head.encode("utf-8")) + len(types_text.encode("utf-8")) + 1
    for mnemonic, block in order:
        size = len(block.encode("utf-8")) + 1
        if used + size > limit:
            cut.append(mnemonic)
            continue
        kept.append(block)
        used += size
    entries = {f"ttkernel_ops.{LANGUAGE}": head + "\n" + "\n".join(kept), f"ttkernel_types.{LANGUAGE}": types_text}
    pack = {f"{key}.txt": text.encode("utf-8") for key, text in entries.items()}
    pack["MANIFEST.yaml"] = _pack_manifest(pin, read, entries, cut, limit)
    example = {f"{EXAMPLE_NAME}/{Path(path).name}": read[path] for path in EXAMPLE_FILES}
    example["MANIFEST.yaml"] = _examples_manifest(pin, example)
    return {f"context/{PACK}": pack, f"examples/{EXAMPLES}": example}


def _digest(data: bytes) -> str:
    """Return the sha256 hex digest of `data`."""
    return hashlib.sha256(data).hexdigest()


def _pack_manifest(
    pin: Mapping[str, str], read: Mapping[str, bytes], entries: Mapping[str, str], cut: list[str], limit: int
) -> bytes:
    """Return the context pack's MANIFEST.yaml."""
    sources = {f"ttkernel_ops.{LANGUAGE}": (OPS_SOURCE,), f"ttkernel_types.{LANGUAGE}": TYPE_SOURCES}
    document = {
        "generator": GENERATOR, "repo": pin["URL"], "commit": pin["COMMIT"], "language": LANGUAGE,
        "limit_bytes": limit,
        "entries": [
            {"key": key, "sources": [{"path": path, "sha256": _digest(read[path])} for path in sources[key]],
             "sha256": _digest(text.encode("utf-8"))}
            for key, text in entries.items()
        ],
        "cut_by_limit": cut,
    }
    head = (
        f"# Generated by {GENERATOR} from tt-mlir at {pin['COMMIT']} (toolchains/tt-mlir.pin); do not edit.\n"
        "# Only this manifest is tracked; the fragments beside it are tt-mlir's text and are gitignored (OQ-018).\n"
        "# No source is a tt-pairs-v0 item (Agent Rule 5). cut_by_limit names, in fill order, every op skipped as it\n"
        "# did not fit limit_bytes.\n"
    )
    return (head + yaml.safe_dump(document, sort_keys=False, width=120)).encode("ascii")


def _examples_manifest(pin: Mapping[str, str], files: Mapping[str, bytes]) -> bytes:
    """Return the worked examples pack's MANIFEST.yaml (lassi.prompts.examples MANIFEST_KEYS)."""
    entries = [{"path": path, "sha256": _digest(data), "bytes": len(data)} for path, data in files.items()]
    document = {
        "generator": GENERATOR, "pack": EXAMPLES, "repo": pin["URL"], "commit": pin["COMMIT"],
        "sources": list(EXAMPLE_FILES),
        "examples": [{"name": EXAMPLE_NAME, "files": entries}],
        "sha256": examples_pack.pack_sha256([(entry["path"], entry["sha256"]) for entry in entries]),
    }
    head = (
        f"# Generated by {GENERATOR} from tt-mlir's own lit tests at {pin['COMMIT']}; do not edit. Only this\n"
        "# manifest is tracked; the files are tt-mlir's text and are gitignored (OQ-018). No file is a tt-pairs-v0\n"
        "# item (Agent Rule 5). sha256 is the sha256 of the lines '<sha256>  <path>' of every file in order.\n"
    )
    return (head + yaml.safe_dump(document, sort_keys=False, width=120)).encode("ascii")


def write(built: Mapping[str, Mapping[str, bytes]], out: Path) -> None:
    """Write every built file under `out`."""
    for tree, files in built.items():
        for path, data in files.items():
            target = out / tree / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)


def compare(built: Mapping[str, Mapping[str, bytes]], out: Path) -> list[str]:
    """Return one message per built file that is missing under `out` or differs from it."""
    problems = []
    for tree, files in built.items():
        for path, data in files.items():
            target = out / tree / path
            if not target.is_file():
                problems.append(f"{target} is missing")
            elif target.read_bytes() != data:
                problems.append(f"{target} differs from what the tree builds")
    return problems


if __name__ == "__main__":
    raise SystemExit(main())
