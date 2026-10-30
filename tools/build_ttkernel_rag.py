"""Build a tt-mlir tests retrieval index from the pinned tt-mlir tree (tasks P17.44 and P17.47).

Usage:

    uv run tools/build_ttkernel_rag.py fetch --dest <dir> [--dialect D]
    uv run tools/build_ttkernel_rag.py build --tree <dir> [--dialect D] [--bench-root <dir>] [--index <file>]
        [--manifest <file>]
    uv run tools/build_ttkernel_rag.py check --tree <dir> [--dialect D] [--bench-root <dir>] [--index <file>]
        [--manifest <file>]

D is a dialect of CORPORA: ttkernel (the default, task P17.44), ttir, or d2m
(task P17.47), each with its corpus spec, assets/rag/<corpus>/corpus.yaml
(read by lassi.prompts.rag_corpus load_corpus_spec, as tools/build_tt_rag.py
reads its own; --corpus names another). A spec selects one dialect's MLIR
from tt-mlir's own lit tests at the pin (toolchains/tt-mlir.pin): modules
that ttmlir-opt parses, verifies, and lowers, the exact textual form of each
op in use. tt-mlir-tests holds the ttkernel tests (the dialect's own tests,
its lowering to EmitC, and its translation to C++); tt-mlir-ttir-tests the
TTIR tests (Dialect/TTIR and the TTIR-to-D2M conversion, whose input is
TTIR); tt-mlir-d2m-tests the D2M tests (Dialect/D2M and the D2M-to-TTKernel
and D2M-to-TTMetal conversions, whose input is D2M). A recipe names the
corpus in its retrieval section, and the runner loads it as it loads the
tt-metal documentation index (lassi.core.runner _retrieval_index): the
manifest beside the spec, and the index under
$LASSI_SCRATCH/rag/<corpus>@<VERSION>/index.json. The index format, the
chunking, BM25, and the manifest's shape are lassi.prompts.retrieval's and
lassi.prompts.rag_corpus's, unchanged.

The tree is tt-mlir at the pin's COMMIT: an install's src/tt-mlir (whose
install's lassi-install.txt names the commit;
tools/build_ttkernel_context.py check_tree), or a directory `fetch` filled
for that corpus (tools/build_tt_rag.py fetch with its spec, each file
checked against its git blob sha1). Any other tree is refused.

Agent Rule 5, enforced here (references, identity_exclusions, comment_runs) and recorded
in the manifest: no item of the spec's suite (tt-pairs-v0-ttkernel, whose
items and answers are tt-pairs-v0's) may reach a prompt through any index.

1. References: every upstream kernel of every item (the C++ answer the
   model translates, read from the suite's fetched sources, `--bench-root`,
   default lassi.bench sources_dir under $LASSI_SCRATCH, each checked
   against its sha256 in the suite manifest when it has one) and every
   tracked file of the suite that exists in this repository. A missing
   upstream kernel refuses the build, so the check never runs without
   them. lassi.prompts.rag_corpus build_corpus then applies its run check
   and similarity report against them, as for the tt-metal documentation.
2. Identities: a selected file whose path or text names an item, an item's
   example directory, or an upstream kernel's file stem (of at least
   MIN_NAME characters) as a whole word, a module that could be derived
   from an item's kernel, is excluded whole, with the name and the item as
   its reason.
3. Comment runs: a selected file holding a reference run once each line
   loses its `//` and FileCheck directive (comment_runs; the C++ a test's
   CHECK lines expect) is excluded whole.
4. The spec's exclude list: for tt-mlir-tests the negative tests, whose IR
   is invalid on purpose, and the worked examples the examples pack already
   carries.

Not Rule 5, but the same whole-file exclusion: a negative test or a test lit
does not run (negative_exclusions: a file with an expected diagnostic, a
`RUN: not` line, or an UNSUPPORTED or XFAIL directive), whose modules are
not known to parse at the pin, unless the spec lists it with its own reason.

`build` writes the index (default under $LASSI_SCRATCH, never in this
repository) and the manifest beside the spec; `check` builds in memory and
compares (status 1 on a difference, 2 on a refusal), as build_tt_rag.py
does. The same tree, spec, and references always give the same bytes.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from lassi.bench import Suite, load_suite  # noqa: E402
from lassi.bench.registry import sources_dir  # noqa: E402
from lassi.prompts import rag_corpus  # noqa: E402
from lassi.prompts.retrieval import index_bytes  # noqa: E402
from lassi.toolchains import pins  # noqa: E402

GENERATOR = "tools/build_ttkernel_rag.py"
# Each dialect's corpus, a directory under assets/rag holding its spec and manifest.
CORPORA = {"ttkernel": "tt-mlir-tests", "ttir": "tt-mlir-ttir-tests", "d2m": "tt-mlir-d2m-tests"}
SPEC = REPO / "assets" / "rag" / CORPORA["ttkernel"] / "corpus.yaml"
# A kernel file stem shorter than this (such as mm) is too common a word to mark an item; the run check still
# covers that kernel's text [DESIGN].
MIN_NAME = 5
# What leaves a line before the comment run check (comment_runs): a leading `//` and a FileCheck directive.
_COMMENT_HEAD = re.compile(r"^\s*//+\s*(?:[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)*:\s*)?")
# A line the comment run check reads (comment_runs): one holding a letter or digit.
_TEXT = re.compile(r"[A-Za-z0-9]")
# What marks a negative test (an expected diagnostic, or a RUN line that expects failure) and a test lit does not run.
_NEGATIVE = re.compile(r"expected-(?:error|warning)|-verify-diagnostics|^//\s*RUN:\s*not\s", re.MULTILINE)
_NOT_RUN = re.compile(r"^//\s*(?P<directive>UNSUPPORTED|XFAIL):", re.MULTILINE)
MANIFEST_HEADER = (
    f"# Generated by {GENERATOR} from the corpus spec beside it and the pinned tt-mlir tree; do not edit.\n"
    "# Only this manifest and corpus.yaml are tracked. The index it pins (index.sha256) is generated under\n"
    "# $LASSI_SCRATCH or a path given, never in the repository, and holds tt-mlir's test text, which stays out of\n"
    "# git as the context packs' text does (OQ-018). This file holds paths, digests, counts, and rules only.\n"
)


def rag_tool() -> ModuleType:
    """Return tools/build_tt_rag.py, whose fetch, file selection, and index check this tool shares."""
    return _tool("build_tt_rag")


def context_tool() -> ModuleType:
    """Return tools/build_ttkernel_context.py, whose check of a tt-mlir install's tree this tool shares."""
    return _tool("build_ttkernel_context")


def _tool(name: str) -> ModuleType:
    """Return tools/<name>.py as a module, loaded once."""
    if name in sys.modules:
        return sys.modules[name]
    path = REPO / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    """Run one subcommand; 0 on success, 1 when check finds a difference, 2 on a refusal."""
    args = _parser().parse_args(argv)
    try:
        spec_path = Path(args.corpus or spec_of(args.dialect)).resolve()
        spec = rag_corpus.load_corpus_spec(spec_path)
        pin = pins.read_pin(spec.pin)
        if args.command == "fetch":
            count = rag_tool().fetch(Path(args.dest), spec, pin)
            print(f"build_ttkernel_rag: {count} files of {pin['URL']} at {pin['COMMIT']} in {args.dest}")
            return 0
        suite = load_suite(REPO / spec.suite)
        bench = Path(args.bench_root) if args.bench_root else _default_bench(suite)
        document, data = build(Path(args.tree), spec_path, spec, pin, bench)
        manifest_path = Path(args.manifest) if args.manifest else spec_path.parent / "MANIFEST.yaml"
        index_path = Path(args.index) if args.index else _default_index(spec, pin, required=args.command == "build")
        if index_path is not None and (index_path.resolve() == REPO or REPO in index_path.resolve().parents):
            raise ValueError(f"the index holds tt-mlir's test text and must lie outside the repository, not at "
                             f"{index_path}")
    except (OSError, ValueError) as error:
        print(f"build_ttkernel_rag: {error}", file=sys.stderr)
        return 2
    if args.command == "check":
        return rag_tool().check(document, manifest_path, index_path)
    _write(index_path, data)
    _write(manifest_path, manifest_text(document).encode("ascii"))
    index = document["index"]
    print(f"build_ttkernel_rag: {index['files']} files, {index['chunks']} chunks, {len(document['excluded'])} "
          f"excluded; index {index_path} sha256 {index['sha256']}")
    for row in document["excluded"]:
        print(f"build_ttkernel_rag: excluded {row['path']}: {row['reason']}")
    return 0


def spec_of(dialect: str) -> Path:
    """Return the corpus spec of a dialect of CORPORA."""
    return REPO / "assets" / "rag" / CORPORA[dialect] / "corpus.yaml"


def _default_bench(suite: Suite) -> Path:
    """Return the suite's fetched sources under $LASSI_SCRATCH; ValueError when it is unset."""
    scratch = os.environ.get("LASSI_SCRATCH")
    if not scratch:
        raise ValueError("LASSI_SCRATCH is not set; set it or pass --bench-root <the suite's fetched sources>")
    return sources_dir(Path(scratch), suite)


def _default_index(spec: rag_corpus.CorpusSpec, pin: Mapping[str, str], *, required: bool) -> Path | None:
    """Return $LASSI_SCRATCH/rag/<corpus>@<VERSION>/index.json; without LASSI_SCRATCH, None (check) or ValueError."""
    if not required and not os.environ.get("LASSI_SCRATCH"):
        return None
    return rag_tool().default_index(spec, pin)


def check_tree(tree: Path, pin: Mapping[str, str], corpus: str) -> None:
    """Raise ValueError unless `tree` is a fetch for `corpus` at the pin's commit or an install's src/tt-mlir."""
    if (tree / rag_tool().FETCH_RECORD).is_file():
        rag_tool().check_tree(tree, pin, corpus)
        return
    context_tool().check_tree(tree, pin)


def references(suite: Suite, bench: Path) -> dict[str, str]:
    """Return {path: text} of every upstream kernel (from `bench`) and every tracked file of `suite` that exists.

    Raises ValueError for an upstream file missing under `bench` or whose
    sha256 differs from the suite manifest's: the Rule 5 check never runs
    without the items' own answers.
    """
    found: dict[str, str] = {}
    for item in suite.items.values():
        for language in item.languages.values():
            for entry in language.support.values():
                path = (REPO / entry.path) if entry.tracked else (bench / entry.path)
                if entry.tracked:
                    if path.is_file():
                        found[entry.path] = path.read_bytes().decode("utf-8")
                    continue
                if not path.is_file():
                    raise ValueError(f"{suite.name}: the upstream file {entry.path} is not under {bench}; fetch the "
                                     f"suite first (uv run tools/fetch_bench.py) or pass --bench-root")
                data = path.read_bytes()
                digest = entry.sha256
                if digest and hashlib.sha256(data).hexdigest() != digest:
                    raise ValueError(f"{path} differs from its sha256 in the {suite.name} manifest")
                found[entry.path] = data.decode("utf-8")
            if language.tracked:
                for name in language.files:
                    path = REPO / language.dir / name
                    if path.is_file():
                        found[f"{language.dir}/{name}"] = path.read_bytes().decode("utf-8")
    return dict(sorted(found.items()))


def identities(suite: Suite) -> dict[str, str]:
    """Return {name: item} of the names that mark an item: its name, example directory, and kernel file stems."""
    names: dict[str, str] = {}
    for item_name, item in sorted(suite.items.items()):
        names.setdefault(item_name, item_name)
        for language in item.languages.values():
            for entry in language.support.values():
                if entry.tracked:
                    continue
                directory = entry.path.partition(f"/{rag_corpus.KERNEL_DIR}/")[0]
                names.setdefault(PurePosixPath(directory).name, item_name)
                stem = PurePosixPath(entry.path).stem
                if len(stem) >= MIN_NAME:
                    names.setdefault(stem, item_name)
    return dict(sorted(names.items()))


def identity_exclusions(files: Mapping[str, bytes], names: Mapping[str, str]) -> dict[str, str]:
    """Return {path: reason} for each file whose path or text names one of `names` as a whole word."""
    found: dict[str, str] = {}
    for path, data in sorted(files.items()):
        text = f"{path}\n{data.decode('utf-8', errors='replace')}"
        for name, item in names.items():
            if re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", text):
                found[path] = f"tier-a-name: names {name}, of the item {item} (Agent Rule 5)"
                break
    return found


def negative_exclusions(files: Mapping[str, bytes]) -> dict[str, str]:
    """Return {path: reason} for each negative test and each test lit does not run (the module docstring).

    A negative test's modules are invalid on purpose; an UNSUPPORTED or
    XFAIL test is not run at the pin, so its modules are not known to parse.
    """
    found: dict[str, str] = {}
    for path, data in sorted(files.items()):
        text = data.decode("utf-8", errors="replace")
        skipped = _NOT_RUN.search(text)
        if skipped is not None:
            found[path] = (f"not-run: an {skipped.group('directive')} test, which lit does not run at the pin, so "
                           "its modules are not known to parse")
        elif _NEGATIVE.search(text):
            found[path] = "negative-test: its modules are invalid on purpose (each with its expected error)"
    return found


def comment_runs(files: Mapping[str, bytes], refs: Mapping[str, str], run_lines: int) -> dict[str, str]:
    """Return {path: reason} for each file holding a reference run in its comments or FileCheck lines.

    Each line loses a leading `//` and a FileCheck directive (CHECK:,
    CHECK-NEXT:, and the like; _COMMENT_HEAD) before rag_corpus's run check
    reads it, so C++ an item's kernel lowers to, quoted in a test's CHECK
    lines, is caught as a copy, which the run check of the lines as written
    misses; such a file goes whole. A line with no letter or digit left (a
    closing brace) is skipped, as the code normalization skips a brace-only
    line: three closing braces in a row are no copy, and before this rule
    they excluded eleven tests whole, ten of D2M and one of TTIR (task P17.47).
    """
    check = rag_corpus.RunCheck(refs, run_lines)
    found: dict[str, str] = {}
    for path, data in sorted(files.items()):
        lines = rag_corpus.split_lines(data.decode("utf-8", errors="replace"))
        stripped = [(number, _COMMENT_HEAD.sub("", line)) for number, line in enumerate(lines, 1)]
        hits = check.matches([(number, line) for number, line in stripped if _TEXT.search(line)]).lines()
        if hits:
            first = min(hits)
            found[path] = (f"tier-a-run in its comments: {len(hits)} line(s) of a reference run, first line {first} "
                           f"from {hits[first]} (Agent Rule 5)")
    return found


def build(tree: Path, spec_path: Path, spec: rag_corpus.CorpusSpec, pin: Mapping[str, str],
          bench: Path) -> tuple[dict[str, Any], bytes]:
    """Return (manifest document, index bytes) for `tree` (the module docstring)."""
    check_tree(tree, pin, spec.corpus)
    suite = load_suite(REPO / spec.suite)
    files = rag_tool().tree_files(tree, spec)
    if not files:
        raise ValueError(f"{tree} holds none of the files {spec_path.name} selects")
    refs = references(suite, bench)
    named = {**negative_exclusions(files), **comment_runs(files, refs, spec.run_lines),
             **identity_exclusions(files, identities(suite))}
    spec = dataclasses.replace(spec, exclude={**spec.exclude, **{k: v for k, v in sorted(named.items())
                                                                 if k not in spec.exclude}})
    tier_dirs = rag_corpus.tier_a_dirs(suite, spec.examples)
    built = rag_corpus.build_corpus(files, spec, tier_dirs=tier_dirs, references=refs)
    spec_sha256 = hashlib.sha256(spec_path.read_bytes()).hexdigest()
    meta = {
        "corpus": spec.corpus,
        "corpus_sha256": spec_sha256,
        "source": {"repo": pin["URL"], "commit": pin["COMMIT"]},
        "chunking": {"max_lines": spec.max_lines, "max_chars": spec.max_chars, "min_piece_lines": spec.min_piece_lines},
        "bm25": {"k1": spec.k1, "b": spec.b},
    }
    data = index_bytes(built.chunks, meta)
    document = rag_corpus.manifest_document(
        built, spec, spec_sha256=spec_sha256, source={"pin": spec.pin, "repo": pin["URL"], "commit": pin["COMMIT"]},
        tier_dirs=tier_dirs, tracked=refs, index_sha256=hashlib.sha256(data).hexdigest(), index_size=len(data))
    document["generator"] = GENERATOR
    return json.loads(json.dumps(document)), data


def manifest_text(document: Mapping[str, Any]) -> str:
    """Return the manifest's YAML text: MANIFEST_HEADER, then the document as rag_corpus writes one."""
    return MANIFEST_HEADER + rag_corpus.manifest_text(document).removeprefix(rag_corpus.MANIFEST_HEADER)


def _write(path: Path, data: bytes) -> None:
    """Write `data` to `path` through a sibling temporary file, so a reader never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_bytes(data)
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(prog="build_ttkernel_rag", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    fetch_parser = commands.add_parser("fetch", help="download the spec's files at the pin's commit")
    fetch_parser.add_argument("--dest", required=True, help="directory outside the repository")
    for name in ("build", "check"):
        command = commands.add_parser(name, help=f"{name} the index and manifest")
        command.add_argument("--tree", required=True, help="the pinned tt-mlir tree (an install's src/tt-mlir or a "
                                                           "fetch)")
        command.add_argument("--bench-root", help="the suite's fetched sources (default under $LASSI_SCRATCH)")
        command.add_argument("--index", help="index file (default $LASSI_SCRATCH/rag/<corpus>@<VERSION>/index.json)")
        command.add_argument("--manifest", help="manifest file (default MANIFEST.yaml beside the corpus spec)")
    for command in commands.choices.values():
        command.add_argument("--dialect", choices=sorted(CORPORA), default="ttkernel",
                             help="the dialect whose corpus spec (CORPORA) is read")
        command.add_argument("--corpus", help="a corpus spec in place of the dialect's")
    return parser


if __name__ == "__main__":
    sys.exit(main())
