"""The documentation corpus of a retrieval index: selection, chunking, and the Agent Rule 5 exclusion.

tools/build_tt_rag.py reads a corpus spec (assets/rag/<corpus>/corpus.yaml),
selects the spec's files from a pinned tree, and calls build_corpus, which
cuts them into chunks (lassi.prompts.retrieval Chunk) and leaves out every
file and chunk Agent Rule 5 forbids: no eval or unassigned item's code may
reach a prompt through retrieval. For the Tier A suite the spec names:

1. Every file under an item's example directory (the directory above
   `kernels/` in each upstream kernel path the suite lists) is excluded,
   and its text becomes a reference.
2. The item's tracked programs (each language with `tracked: true`) are
   references too; they are read only to be kept out, never indexed.
3. Every path the spec lists under `exclude` is excluded with its reason:
   the documentation pages that walk through a Tier A example, whose prose
   narrates the reference solution even where its code is left out.
4. Run check: a run is `run_lines` consecutive lines of a file's chunked
   lines under one of two normalizations, applied alike to the references
   and to the corpus, and a reference run is one of a reference under the
   same normalization:
   - code (normalized_lines): each line's code alone. A block comment's
     close with no opening before it ends a comment begun above, so the
     line's text up to it goes; every /* */ span goes; an unclosed /*
     and a // comment end the line; whitespace is stripped and collapsed.
     A line left empty (blank, comment only, or between comment markers),
     starting with * (a block comment's middle line), or holding braces
     with only parentheses, semicolons, and commas beside them is skipped.
     A comment between copied lines, or one added or dropped beside them,
     cannot hide a run of reference code.
   - literal (literal_lines): each non-blank line stripped with inner
     whitespace collapsed, comments and braces kept, so copied comments
     and prose are caught too.
   Each line normalizes on its own, never by its neighbours, so the lines
   of any chunk, or any piece of one, normalize as they do in the file.
   A file more than `max_file_overlap` of whose runs under either
   normalization are reference runs is excluded whole, as a copy of
   reference code. In any other file every line of every reference run of
   either normalization is excised: left out of every chunk, each chunk cut
   around it, so no indexed line lies in a reference run of its file, and
   no chunk holds a reference run. Only the matching normalizes: a chunk's
   text stays the file's lines exactly, comments included. Every literal
   run is counted, a run of closing braces too: the rule stays literal, and
   an excised brace costs nothing.
5. Pieces: of a chunk an excision cut, a piece is dropped, and recorded as
   an excision with its reason, when it holds fewer than `min_piece_lines`
   non-blank lines (short-piece), no letter or digit (no-text), or a
   Markdown code fence whose pair lies in the same chunk but outside the
   piece (unpaired-fence). A chunk no excision touched stays whole.
6. Similarity report: every indexed code file whose indexed text has a
   token similarity (similarity_tokens, difflib's ratio) of at least
   `report_at` to a Tier A code reference is recorded with its closest
   reference. It catches a renamed or reflowed copy that no run shows; the
   build records it and never refuses on it.

Chunking: Markdown and reStructuredText files are cut at headings into
sections, code files (.hpp, .h, .cpp, .tpp) into blocks at blank lines; a
fenced or literal code block is never cut at a blank line. Blocks are packed
in order into chunks of at most `max_lines` lines and `max_chars`
characters; a longer block is cut into pieces of whole lines, and a line is
never cut. A code file's leading license comment (a comment block holding
SPDX-) is left out of every chunk, and so is a chunk with no letter or
digit. Lines are counted at "\\n", a trailing "\\r" dropped, so a chunk's
range matches the file's line numbers and its text is those lines exactly.
"""

from __future__ import annotations

import difflib
import hashlib
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from lassi.bench.registry import Suite
from lassi.prompts.retrieval import GENERATOR, INDEX_FORMAT, TOKENIZER, Chunk

# The keys of a corpus spec, and of its tier_a, chunking, and bm25 sections.
SPEC_KEYS = {"corpus", "pin", "include", "doxygen", "tier_a", "exclude", "chunking", "bm25"}
TIER_A_KEYS = {"suite", "examples", "run_lines", "max_file_overlap", "similarity"}
SIMILARITY_KEYS = {"report_at", "api_prefixes"}
CHUNKING_KEYS = {"max_lines", "max_chars", "min_piece_lines"}
# The keys of a manifest (manifest_document), and the comment every manifest opens with.
MANIFEST_KEYS = {"corpus", "generator", "corpus_sha256", "source", "index", "chunking", "bm25", "tier_a", "files",
                 "excluded", "excised", "similar"}
MANIFEST_HEADER = (
    f"# Generated by {GENERATOR} from the corpus spec beside it and the pinned tree; do not edit.\n"
    "# Only this manifest and corpus.yaml are tracked. The index it pins (index.sha256) is generated under\n"
    "# $LASSI_SCRATCH or a path given, never in the repository, and holds the corpus text, which stays out of git\n"
    "# as the context packs' text does (OQ-018). This file holds paths, digests, counts, and rules only.\n"
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
# The directory that marks a kernel in an upstream support path, the rule lassi.toolchains.ttmetal_build KERNEL_DIR
# names kernel sources by; the example directory is its parent.
KERNEL_DIR = "kernels"
# File kinds by suffix; every other suffix is code.
MARKDOWN = {".md"}
RST = {".rst"}

_FENCE = re.compile(r"\s*(```|~~~)")
_MD_HEADING = re.compile(r"(#{1,6})\s+(\S.*?)\s*#*\s*$")
_RST_ADORNMENT = re.compile(r"([=\-~^*#\"'`:.+_])\1{2,}\s*$")
_RST_LITERAL = re.compile(r"\s*\.\.\s+(code-block|code|sourcecode|highlight)::|.*::\s*$")
_INDENT = re.compile(r"[ \t]*")
_DOXY_INPUT = re.compile(r"INPUT\s*\+?=(.*)")
_ALNUM = re.compile(r"[A-Za-z0-9]")
# The code normalization (code_line): a /* */ span, and a line of braces with only parentheses, semicolons, commas.
_BLOCK_SPAN = re.compile(r"/\*.*?\*/")
_BRACES_ONLY = re.compile(r"[{}();,\s]*")
# The similarity report's tokens (similarity_tokens): a name, a number, a string literal, or one other character; the
# lower-case names that stay themselves (C++ keywords and types); comments; and an #include line.
_SIM_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|\"[^\"\n]*\"|\S")
_SIM_NAME = re.compile(r"[a-z_][a-z0-9_]*")
_SIM_DIGIT = re.compile(r"\d")
_SIM_KEYWORDS = frozenset("auto const constexpr uint32_t int float for if else return void bool".split())
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)
_LINE_COMMENT = re.compile(r"//[^\n]*")


@dataclass(frozen=True)
class CorpusSpec:
    """A checked corpus spec (the module docstring); every field is required."""

    corpus: str
    pin: str
    include: tuple[str, ...]
    doxygen: str | None
    suite: str
    examples: str
    run_lines: int
    max_file_overlap: float
    report_at: float
    api_prefixes: tuple[str, ...]
    exclude: Mapping[str, str]
    max_lines: int
    max_chars: int
    min_piece_lines: int
    k1: float
    b: float


@dataclass(frozen=True)
class IndexedFile:
    """A file of the index: its path, the sha256 of its bytes, and how many of its chunks are indexed."""

    path: str
    sha256: str
    chunks: int


@dataclass(frozen=True)
class ExcludedFile:
    """A selected file left out whole: its path, the sha256 of its bytes, and the rule that left it out."""

    path: str
    sha256: str
    reason: str


@dataclass(frozen=True)
class ExcisedLines:
    """Lines the run check left out of every chunk: their file, first and last line, and the reference run's file."""

    path: str
    start_line: int
    end_line: int
    reason: str


@dataclass(frozen=True)
class RunMatches:
    """What RunCheck.matches found in one text, per normalization (NORMALIZATIONS order).

    `matched` and `total` count the runs that are reference runs and the
    runs in all; `hits` maps each line of a matched run to its reference.
    """

    matched: Mapping[str, int]
    total: Mapping[str, int]
    hits: Mapping[str, Mapping[int, str]]

    def lines(self) -> dict[int, str]:
        """Return {line number: reference} for every line of a matched run of any normalization, code first."""
        found: dict[int, str] = {}
        for kind in NORMALIZATIONS:
            for number, reference in self.hits[kind].items():
                found.setdefault(number, reference)
        return found

    def overlap(self, limit: float) -> str | None:
        """Return the tier-a-overlap reason when more than `limit` of one normalization's runs matched, else None."""
        for kind in NORMALIZATIONS:
            matched, total, hits = self.matched[kind], self.total[kind], self.hits[kind]
            if total and matched / total > limit:
                first = hits[min(hits)]
                return f"tier-a-overlap: {matched} of its {total} {kind} runs are reference runs, first in {first}"
        return None


@dataclass(frozen=True)
class SimilarFile:
    """An indexed code file at or above the report line: its similarity (3 decimals) and its closest reference."""

    path: str
    similarity: float
    reference: str


@dataclass(frozen=True)
class CorpusBuild:
    """What build_corpus made: the chunks to index, the record of every file and excised line, and the report."""

    chunks: tuple[Chunk, ...]
    files: tuple[IndexedFile, ...]
    excluded: tuple[ExcludedFile, ...]
    excised: tuple[ExcisedLines, ...]
    similar: tuple[SimilarFile, ...]


def load_corpus_spec(path: Path | str) -> CorpusSpec:
    """Read and check a corpus spec; any problem raises ValueError naming the file and the key."""
    path = Path(path)
    data = yaml.safe_load(path.read_bytes().decode("utf-8"))
    where = str(path)
    _exact_keys(data, SPEC_KEYS, where)
    tier_a, chunking, bm25 = data["tier_a"], data["chunking"], data["bm25"]
    _exact_keys(tier_a, TIER_A_KEYS, f"{where}: tier_a")
    similarity = tier_a["similarity"]
    _exact_keys(similarity, SIMILARITY_KEYS, f"{where}: tier_a.similarity")
    prefixes = similarity["api_prefixes"]
    if not isinstance(prefixes, list) or not all(isinstance(p, str) and p for p in prefixes):
        raise ValueError(f"{where}: tier_a.similarity.api_prefixes must be a list of non-empty strings")
    _exact_keys(chunking, CHUNKING_KEYS, f"{where}: chunking")
    _exact_keys(bm25, {"k1", "b"}, f"{where}: bm25")
    include = data["include"]
    if not isinstance(include, list) or not include or not all(isinstance(p, str) and p for p in include):
        raise ValueError(f"{where}: include must be a non-empty list of glob strings")
    exclude = data["exclude"] or []
    if not isinstance(exclude, list) or not all(
            isinstance(e, dict) and set(e) == {"path", "reason"} and all(isinstance(v, str) and v for v in e.values())
            for e in exclude):
        raise ValueError(f"{where}: exclude must be a list of {{path, reason}} entries")
    for key, value in (("corpus", data["corpus"]), ("pin", data["pin"]), ("tier_a.suite", tier_a["suite"]),
                       ("tier_a.examples", tier_a["examples"])):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{where}: {key} must be a non-empty string")
    if data["doxygen"] is not None and not isinstance(data["doxygen"], str):
        raise ValueError(f"{where}: doxygen must be a path in the tree or null")
    return CorpusSpec(
        corpus=data["corpus"], pin=data["pin"], include=tuple(include), doxygen=data["doxygen"],
        suite=tier_a["suite"], examples=tier_a["examples"].strip("/"),
        run_lines=_positive(tier_a["run_lines"], f"{where}: tier_a.run_lines"),
        max_file_overlap=_fraction(tier_a["max_file_overlap"], f"{where}: tier_a.max_file_overlap"),
        report_at=_fraction(similarity["report_at"], f"{where}: tier_a.similarity.report_at"),
        api_prefixes=tuple(prefixes),
        exclude={entry["path"]: entry["reason"] for entry in exclude},
        max_lines=_positive(chunking["max_lines"], f"{where}: chunking.max_lines"),
        max_chars=_positive(chunking["max_chars"], f"{where}: chunking.max_chars"),
        min_piece_lines=_positive(chunking["min_piece_lines"], f"{where}: chunking.min_piece_lines"),
        k1=_number(bm25["k1"], f"{where}: bm25.k1"), b=_fraction(bm25["b"], f"{where}: bm25.b"),
    )


def glob_match(pattern: str, path: str) -> bool:
    """Return True when POSIX `path` matches `pattern`: `**/` spans directories, `*` and `?` stay in one."""
    regex = ""
    index = 0
    while index < len(pattern):
        if pattern.startswith("**/", index):
            regex, index = regex + "(?:.*/)?", index + 3
        elif pattern[index] == "*":
            regex, index = regex + "[^/]*", index + 1
        elif pattern[index] == "?":
            regex, index = regex + "[^/]", index + 1
        else:
            regex, index = regex + re.escape(pattern[index]), index + 1
    return re.fullmatch(regex, path) is not None


def doxygen_inputs(text: str) -> list[str]:
    """Return the paths a Doxyfile's INPUT (and INPUT +=) lines list, each once, in order."""
    found: list[str] = []
    lines = split_lines(text)
    index = 0
    while index < len(lines):
        match = _DOXY_INPUT.match(lines[index].strip())
        index += 1
        if match is None:
            continue
        value = match.group(1)
        while value.rstrip().endswith("\\") and index < len(lines):
            value = value.rstrip()[:-1] + " " + lines[index]
            index += 1
        found.extend(value.rstrip().rstrip("\\").split())
    return list(dict.fromkeys(found))


def split_lines(text: str) -> list[str]:
    """Return the lines of `text` split at "\\n" only, each trailing "\\r" dropped; a final newline ends no line."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return [line[:-1] if line.endswith("\r") else line for line in lines]


def tier_a_dirs(suite: Suite, examples: str) -> dict[str, str]:
    """Return {example directory: item name} for every upstream kernel path of every item of `suite`.

    Raises ValueError for an upstream path with no kernels directory or
    outside the tree directory `examples`. The items are read only to keep
    them out.
    """
    dirs: dict[str, str] = {}
    for name, item in sorted(suite.items.items()):
        for language in item.languages.values():
            for entry in language.support.values():
                if entry.tracked:
                    continue
                head, marker, _ = entry.path.partition(f"/{KERNEL_DIR}/")
                if not marker or not head.startswith(f"{examples}/"):
                    raise ValueError(f"{suite.name}/{name}: upstream path {entry.path} names no example directory "
                                     f"under {examples}/ holding {KERNEL_DIR}/")
                dirs.setdefault(head, name)
    return dict(sorted(dirs.items()))


def tracked_references(suite: Suite, root: Path) -> dict[str, str]:
    """Return {repository path: text} of every tracked program of every item, read under `root`."""
    found: dict[str, str] = {}
    for item in suite.items.values():
        for language in item.languages.values():
            if language.tracked:
                for name in language.files:
                    path = f"{language.dir}/{name}"
                    found[path] = (Path(root) / path).read_bytes().decode("utf-8")
    return dict(sorted(found.items()))


def chunk_file(path: str, text: str, *, max_lines: int, max_chars: int) -> list[Chunk]:
    """Return the chunks of one file in line order (the module docstring).

    Every non-blank line is in exactly one chunk, except a code file's
    license comment and a chunk with no letter or digit, which are left out.
    """
    lines = split_lines(text)
    suffix = Path(path).suffix.lower()
    if suffix in MARKDOWN:
        blocks = _markdown_blocks(lines)
    elif suffix in RST:
        blocks = _rst_blocks(lines)
    else:
        blocks = _code_blocks(lines, _license_end(lines))
    chunks: list[Chunk] = []
    for start, end, heading in _pack(blocks, lines, max_lines, max_chars):
        text = "\n".join(lines[start:end + 1])
        if _ALNUM.search(text):
            chunks.append(Chunk(path, start + 1, end + 1, heading, text))
    return chunks


class RunCheck:
    """The Rule 5 run check: which lines of a text lie in a run of `run_lines` lines of a reference."""

    def __init__(self, references: Mapping[str, str], run_lines: int) -> None:
        """Index every run of every reference under each normalization, kept with the first reference holding it.

        References are read in path order, so a run two references share
        names the first.
        """
        self.run_lines = run_lines
        self._runs: dict[str, dict[tuple[str, ...], str]] = {kind: {} for kind in NORMALIZATIONS}
        for name in sorted(references):
            lines = split_lines(references[name])
            for kind, normalize in NORMALIZATIONS.items():
                sequence = [norm for norm in map(normalize, lines) if norm]
                for run in _windows(sequence, run_lines):
                    self._runs[kind].setdefault(run, name)

    def matches(self, lines: Sequence[tuple[int, str]]) -> RunMatches:
        """Return the reference runs of `lines`, (line number, line as written) for the lines to check, in order.

        Each normalization reads every line on its own and skips the lines
        it leaves empty, so a run may span blank and skipped lines.
        """
        matched: dict[str, int] = {}
        total: dict[str, int] = {}
        hits: dict[str, dict[int, str]] = {}
        for kind, normalize in NORMALIZATIONS.items():
            numbered = [(number, norm) for number, line in lines if (norm := normalize(line))]
            windows = _windows([norm for _, norm in numbered], self.run_lines)
            matched[kind], total[kind], hits[kind] = 0, len(windows), {}
            for offset, run in enumerate(windows):
                reference = self._runs[kind].get(run)
                if reference is not None:
                    matched[kind] += 1
                    for number, _ in numbered[offset:offset + self.run_lines]:
                        hits[kind].setdefault(number, reference)
        return RunMatches(matched, total, hits)


def code_line(line: str) -> str:
    """Return the code of one line for the run check, or '' when the line holds none (the module docstring, 4).

    A block comment's close with no /* or // before it ends a comment begun
    above, so the text up to it goes; then every /* */ span goes, and an
    unclosed /* or a // ends the line. The rest is stripped with inner
    whitespace collapsed; it counts as no code when it starts with * (a
    block comment's middle line) or holds braces with only parentheses,
    semicolons, and commas beside them.
    """
    close = line.find("*/")
    if close >= 0 and "/*" not in line[:close] and "//" not in line[:close]:
        line = line[close + 2:]
    line = _BLOCK_SPAN.sub(" ", line).split("/*", 1)[0].split("//", 1)[0]
    text = " ".join(line.split())
    if text.startswith("*") or (("{" in text or "}" in text) and _BRACES_ONLY.fullmatch(text)):
        return ""
    return text


def literal_line(line: str) -> str:
    """Return one line stripped with inner whitespace collapsed, comments and braces kept ('' when blank)."""
    return " ".join(line.split())


# The run check's normalizations by name, code first: the names the manifest's tier-a-overlap reasons give.
NORMALIZATIONS: dict[str, Callable[[str], str]] = {"code": code_line, "literal": literal_line}


def normalized_lines(lines: Sequence[str], first: int = 1) -> list[tuple[int, str]]:
    """Return (line number, code_line) for each line holding code: comments, blank, and brace-only lines skipped."""
    return [(first + offset, norm) for offset, norm in enumerate(map(code_line, lines)) if norm]


def literal_lines(lines: Sequence[str], first: int = 1) -> list[tuple[int, str]]:
    """Return (line number, literal_line) for each non-blank line: the run check's literal normalization."""
    return [(first + offset, norm) for offset, norm in enumerate(map(literal_line, lines)) if norm]


def build_corpus(files: Mapping[str, bytes], spec: CorpusSpec, *, tier_dirs: Mapping[str, str],
                 references: Mapping[str, str]) -> CorpusBuild:
    """Chunk the selected `files` ({tree path: bytes}) and apply the Rule 5 exclusions (the module docstring).

    `tier_dirs` maps each Tier A example directory to its item; `references`
    holds the tracked programs. Files are processed in path order, so the
    result depends only on the arguments.
    """
    digests = {path: _sha256(data) for path, data in files.items()}
    all_refs = dict(references)
    excluded: list[ExcludedFile] = []
    candidates: list[tuple[str, str]] = []
    for path in sorted(files):
        item = next((name for directory, name in tier_dirs.items() if path.startswith(f"{directory}/")), None)
        text = _decoded(files[path])
        if item is not None:
            excluded.append(ExcludedFile(path, digests[path], f"tier-a-item: {item}"))
            if text is not None:
                all_refs[path] = text
        elif path in spec.exclude:
            excluded.append(ExcludedFile(path, digests[path], f"listed: {spec.exclude[path]}"))
        elif text is None:
            excluded.append(ExcludedFile(path, digests[path], "not-utf8"))
        else:
            candidates.append((path, text))
    check = RunCheck(all_refs, spec.run_lines)
    chunks: list[Chunk] = []
    indexed: list[IndexedFile] = []
    excised: list[ExcisedLines] = []
    for path, text in candidates:
        kept, cut, reason = _checked_chunks(path, text, spec, check)
        if reason is not None:
            excluded.append(ExcludedFile(path, digests[path], reason))
            continue
        excised.extend(cut)
        if kept:
            chunks.extend(kept)
            indexed.append(IndexedFile(path, digests[path], len(kept)))
        else:
            excluded.append(ExcludedFile(path, digests[path], "no-chunk"))
    texts: dict[str, list[str]] = {}
    for chunk in chunks:
        if _is_code(chunk.path):
            texts.setdefault(chunk.path, []).append(chunk.text)
    similar = similarity_report({path: "\n".join(parts) for path, parts in texts.items()},
                                {name: text for name, text in all_refs.items() if _is_code(name)},
                                report_at=spec.report_at, api_prefixes=spec.api_prefixes)
    return CorpusBuild(tuple(chunks), tuple(indexed), tuple(sorted(excluded, key=lambda e: e.path)),
                       tuple(excised), tuple(similar))


def similarity_tokens(text: str, api_prefixes: Sequence[str]) -> list[str]:
    """Return the tokens the similarity report compares: a copy that renames its locals keeps the same tokens.

    Comments and #include lines go. A token is a name, a number, a string
    literal, or one other character; a lower-case name reads as v unless it
    is a C++ keyword or type of _SIM_KEYWORDS or starts with an API prefix,
    and a number reads as L, so renaming locals or changing constants keeps
    the tokens, while the API calls and the structure stay as they are.
    """
    text = _LINE_COMMENT.sub(" ", _BLOCK_COMMENT.sub(" ", text))
    text = "\n".join(line for line in text.split("\n") if not line.strip().startswith("#include"))
    prefixes = tuple(api_prefixes)
    tokens: list[str] = []
    for token in _SIM_TOKEN.findall(text):
        if _SIM_NAME.fullmatch(token) and token not in _SIM_KEYWORDS and not token.startswith(prefixes):
            token = "v"
        elif _SIM_DIGIT.match(token):
            token = "L"
        tokens.append(token)
    return tokens


def similarity_report(texts: Mapping[str, str], references: Mapping[str, str], *, report_at: float,
                      api_prefixes: Sequence[str]) -> list[SimilarFile]:
    """Return, in path order, every text whose similarity to its closest reference is at least `report_at`.

    The similarity is difflib.SequenceMatcher(None, text, reference,
    autojunk=False).ratio() over similarity_tokens; the closest reference is
    the one with the highest ratio, the first in path order on a tie. A
    reference whose quick upper bounds fall under `report_at` is skipped, as
    its ratio cannot reach it.
    """
    matchers: list[tuple[str, difflib.SequenceMatcher]] = []
    for name in sorted(references):
        matcher = difflib.SequenceMatcher(None, autojunk=False)
        matcher.set_seq2(similarity_tokens(references[name], api_prefixes))
        matchers.append((name, matcher))
    found: list[SimilarFile] = []
    for path in sorted(texts):
        tokens = similarity_tokens(texts[path], api_prefixes)
        best: tuple[float, str] | None = None
        for name, matcher in matchers:
            matcher.set_seq1(tokens)
            if matcher.real_quick_ratio() < report_at or matcher.quick_ratio() < report_at:
                continue
            ratio = matcher.ratio()
            if ratio >= report_at and (best is None or ratio > best[0]):
                best = (ratio, name)
        if best is not None:
            found.append(SimilarFile(path, round(best[0], 3), best[1]))
    return found


def manifest_document(build: CorpusBuild, spec: CorpusSpec, *, spec_sha256: str, source: Mapping[str, str],
                      tier_dirs: Mapping[str, str], tracked: Mapping[str, str], index_sha256: str,
                      index_size: int) -> dict[str, Any]:
    """Return the manifest of one build: the source pin, every file's sha256, the parameters, and the index's sha256.

    `source` is {pin, repo, commit}; `tracked` the tracked references
    ({repository path: text}), recorded by sha256 so a changed reference
    makes the manifest stale. The manifest holds paths, digests, counts, and
    rules only, never a line of the corpus.
    """
    return {
        "corpus": spec.corpus,
        "generator": GENERATOR,
        "corpus_sha256": spec_sha256,
        "source": {key: source[key] for key in ("pin", "repo", "commit")},
        "index": {"format": INDEX_FORMAT, "tokenizer": TOKENIZER, "sha256": index_sha256, "bytes": index_size,
                  "files": len(build.files), "chunks": len(build.chunks)},
        "chunking": {"max_lines": spec.max_lines, "max_chars": spec.max_chars,
                     "min_piece_lines": spec.min_piece_lines},
        "bm25": {"k1": spec.k1, "b": spec.b},
        "tier_a": {
            "suite": spec.suite, "examples": spec.examples, "run_lines": spec.run_lines,
            "max_file_overlap": spec.max_file_overlap,
            "similarity": {"report_at": spec.report_at, "api_prefixes": list(spec.api_prefixes)},
            "dirs": [{"path": path, "item": item} for path, item in sorted(tier_dirs.items())],
            "references": [{"path": path, "sha256": _sha256(text.encode("utf-8"))}
                           for path, text in sorted(tracked.items())],
        },
        "files": [{"path": f.path, "sha256": f.sha256, "chunks": f.chunks} for f in build.files],
        "excluded": [{"path": e.path, "sha256": e.sha256, "reason": e.reason} for e in build.excluded],
        "excised": [{"path": x.path, "first": x.start_line, "last": x.end_line, "reason": x.reason}
                    for x in build.excised],
        "similar": [{"path": s.path, "similarity": s.similarity, "reference": s.reference} for s in build.similar],
    }


def manifest_text(document: Mapping[str, Any]) -> str:
    """Return a manifest's YAML text: MANIFEST_HEADER, then the document, one row per file."""
    body = yaml.safe_dump(dict(document), sort_keys=False, default_flow_style=None, width=1000, allow_unicode=False)
    return MANIFEST_HEADER + body


def read_manifest(path: Path | str) -> dict[str, Any]:
    """Read a manifest and check its shape; raise ValueError naming GENERATOR when it is not one."""
    path = Path(path)
    regenerate = f"regenerate it with uv run {GENERATOR} build"
    try:
        data = yaml.safe_load(path.read_bytes().decode("ascii"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise ValueError(f"cannot read the manifest {path} ({error}); {regenerate}") from error
    if not isinstance(data, dict) or set(data) != MANIFEST_KEYS:
        raise ValueError(f"{path} must hold exactly the keys {', '.join(sorted(MANIFEST_KEYS))}; {regenerate}")
    digests = [data["corpus_sha256"], data["index"].get("sha256")]
    digests += [row.get("sha256") for key in ("files", "excluded") for row in data[key]]
    digests += [row.get("sha256") for row in data["tier_a"].get("references", [])]
    if not all(isinstance(value, str) and _SHA256.fullmatch(value) for value in digests):
        raise ValueError(f"{path} holds a sha256 that is not 64 lower-case hex digits; {regenerate}")
    return data


def _checked_chunks(path: str, text: str, spec: CorpusSpec,
                    check: RunCheck) -> tuple[list[Chunk], list[ExcisedLines], str | None]:
    """Return (chunks with reference runs excised, excised ranges, None), or ([], [], reason) to exclude the file."""
    chunks = chunk_file(path, text, max_lines=spec.max_lines, max_chars=spec.max_chars)
    lines = [(chunk.start_line + offset, line) for chunk in chunks
             for offset, line in enumerate(split_lines(chunk.text))]
    found = check.matches(lines)
    reason = found.overlap(spec.max_file_overlap)
    if reason is not None:
        return [], [], reason
    hits = found.lines()
    fences = _fence_pairs(split_lines(text)) if Path(path).suffix.lower() in MARKDOWN else {}
    kept: list[Chunk] = []
    dropped: list[ExcisedLines] = []
    for chunk in chunks:
        if not any(number in hits for number in range(chunk.start_line, chunk.end_line + 1)):
            kept.append(chunk)
            continue
        for piece in _without(chunk, hits):
            why = _weak_piece(piece, chunk, spec.min_piece_lines, fences)
            if why is None:
                kept.append(piece)
            else:
                dropped.append(ExcisedLines(path, piece.start_line, piece.end_line, why))
    ranges = _run_ranges(path, lines, hits) + dropped
    return kept, sorted(ranges, key=lambda row: (row.start_line, row.end_line)), None


def _run_ranges(path: str, lines: Sequence[tuple[int, str]], hits: Mapping[int, str]) -> list[ExcisedLines]:
    """Return the excised ranges of `hits`: consecutive hit lines, blank lines between them skipped, as one range."""
    ranges: list[ExcisedLines] = []
    previous = False
    for number, line in lines:
        if not line.strip():
            continue
        if number not in hits:
            previous = False
            continue
        if previous:
            last = ranges[-1]
            ranges[-1] = ExcisedLines(path, last.start_line, number, last.reason)
        else:
            ranges.append(ExcisedLines(path, number, number, f"tier-a-run: {hits[number]}"))
        previous = True
    return ranges


def _weak_piece(piece: Chunk, chunk: Chunk, min_lines: int, fences: Mapping[int, int]) -> str | None:
    """Return why a piece an excision left of `chunk` is dropped (the module docstring, 5), or None to keep it."""
    if sum(1 for line in split_lines(piece.text) if line.strip()) < min_lines:
        return f"short-piece: fewer than {min_lines} lines left beside an excision"
    if not _ALNUM.search(piece.text):
        return "no-text: no letter or digit left beside an excision"
    for number in range(piece.start_line, piece.end_line + 1):
        partner = fences.get(number)
        if partner is not None and chunk.start_line <= partner <= chunk.end_line \
                and not piece.start_line <= partner <= piece.end_line:
            return "unpaired-fence: an excision cut a code fence off from its pair"
    return None


def _fence_pairs(lines: Sequence[str]) -> dict[int, int]:
    """Return {line number: its pair's} for every paired Markdown code fence, matched as _markdown_blocks does."""
    pairs: dict[int, int] = {}
    opened: tuple[int, str] | None = None
    for number, line in enumerate(lines, start=1):
        match = _FENCE.match(line)
        if match is None:
            continue
        if opened is None:
            opened = (number, match.group(1))
        elif match.group(1) == opened[1]:
            pairs[opened[0]], pairs[number] = number, opened[0]
            opened = None
    return pairs


def _is_code(path: str) -> bool:
    """Return True for a code file: any suffix but Markdown's and reStructuredText's."""
    return Path(path).suffix.lower() not in MARKDOWN | RST


def _without(chunk: Chunk, hits: Mapping[int, str]) -> list[Chunk]:
    """Return the pieces of `chunk` left when every line in `hits` is cut out; blank ends and empty pieces go."""
    pieces: list[Chunk] = []
    run: list[tuple[int, str]] = []

    def flush() -> None:
        """Close the current piece: trim its blank ends and keep it when any line is left."""
        while run and not run[-1][1].strip():
            run.pop()
        while run and not run[0][1].strip():
            run.pop(0)
        if run:
            pieces.append(Chunk(chunk.path, run[0][0], run[-1][0], chunk.heading, "\n".join(line for _, line in run)))
        run.clear()

    for offset, line in enumerate(split_lines(chunk.text)):
        number = chunk.start_line + offset
        if number in hits:
            flush()
        else:
            run.append((number, line))
    flush()
    return pieces


# A block: first line, last line (0-based, inclusive), the heading path in effect from it ("" when none), and
# is_heading: True when the block is a heading or section title itself, which _pack lets start a new chunk.
_Block = tuple[int, int, str, bool]


def _markdown_blocks(lines: Sequence[str]) -> list[_Block]:
    """Return the blocks of a Markdown file: headings, fenced code blocks, and paragraphs between blank lines."""
    blocks: list[_Block] = []
    stack: list[tuple[int, str]] = []
    fence: str | None = None
    start: int | None = None
    for number, line in enumerate(lines):
        fence_match = _FENCE.match(line)
        heading = _MD_HEADING.match(line) if fence is None else None
        if heading is not None:
            _close(blocks, start, number - 1, lines, _path(stack), False)
            level = len(heading.group(1))
            stack = [entry for entry in stack if entry[0] < level] + [(level, heading.group(2))]
            blocks.append((number, number, _path(stack), True))
            start = None
            continue
        if fence_match is not None:
            marker = fence_match.group(1)
            fence = marker if fence is None else (None if marker == fence else fence)
        if not line.strip() and fence is None:
            _close(blocks, start, number - 1, lines, _path(stack), False)
            start = None
        elif start is None:
            start = number
    _close(blocks, start, len(lines) - 1, lines, _path(stack), False)
    return blocks


def _rst_blocks(lines: Sequence[str]) -> list[_Block]:
    """Return the blocks of a reStructuredText file: section titles, literal blocks, and paragraphs."""
    blocks: list[_Block] = []
    styles: list[tuple[bool, str]] = []
    stack: list[tuple[int, str]] = []
    start: int | None = None
    number = 0
    while number < len(lines):
        title = _rst_title(lines, number)
        if title is not None:
            first, last, style, text = title
            _close(blocks, start, first - 1, lines, _path(stack), False)
            if style not in styles:
                styles.append(style)
            level = styles.index(style)
            stack = [entry for entry in stack if entry[0] < level] + [(level, text)]
            blocks.append((first, last, _path(stack), True))
            start, number = None, last + 1
            continue
        line = lines[number]
        if line.strip() and _RST_LITERAL.match(line):
            if start is None:
                start = number
            number = _literal_end(lines, number) + 1
            continue
        if not line.strip():
            _close(blocks, start, number - 1, lines, _path(stack), False)
            start = None
        elif start is None:
            start = number
        number += 1
    _close(blocks, start, len(lines) - 1, lines, _path(stack), False)
    return blocks


def _rst_title(lines: Sequence[str], number: int) -> tuple[int, int, tuple[bool, str], str] | None:
    """Return (first line, last line, adornment style, title) when a section title starts at `number`."""
    line = lines[number]
    over = _RST_ADORNMENT.match(line)
    if over and number + 2 < len(lines) and lines[number + 1].strip() and not _RST_ADORNMENT.match(lines[number + 1]):
        under = _RST_ADORNMENT.match(lines[number + 2])
        if under and under.group(1) == over.group(1):
            return number, number + 2, (True, over.group(1)), lines[number + 1].strip()
    if line.strip() and not over and not line[0].isspace() and number + 1 < len(lines):
        under = _RST_ADORNMENT.match(lines[number + 1])
        if under and not lines[number + 1][0].isspace() and len(lines[number + 1].rstrip()) >= len(line.rstrip()):
            return number, number + 1, (False, under.group(1)), line.strip()
    return None


def _literal_end(lines: Sequence[str], number: int) -> int:
    """Return the last line of the literal block a `::` or code directive at `number` opens (itself if none)."""
    indent = len(_INDENT.match(lines[number]).group(0))
    last = number
    for later in range(number + 1, len(lines)):
        line = lines[later]
        if not line.strip():
            continue
        if len(_INDENT.match(line).group(0)) <= indent:
            break
        last = later
    return last


def _code_blocks(lines: Sequence[str], first: int) -> list[_Block]:
    """Return the blocks of a code file from line `first`: runs of non-blank lines."""
    blocks: list[_Block] = []
    start: int | None = None
    for number in range(first, len(lines)):
        if not lines[number].strip():
            _close(blocks, start, number - 1, lines, "", False)
            start = None
        elif start is None:
            start = number
    _close(blocks, start, len(lines) - 1, lines, "", False)
    return blocks


def _license_end(lines: Sequence[str]) -> int:
    """Return the first line after a file's leading license comment (0 when no leading comment holds SPDX-).

    The license comment runs from the top through its last SPDX- line and
    the end of the /* */ comment that line sits in, if any, then over bare
    comment markers and blank lines; a doc comment after it stays.
    """
    last: int | None = None
    closes = 0
    in_block = False
    for number, line in enumerate(lines):
        stripped = line.strip()
        if not (in_block or not stripped or stripped.startswith(("//", "/*"))):
            break
        opened = in_block or stripped.startswith("/*")
        in_block = opened and "*/" not in (stripped if in_block else stripped[2:])
        if "SPDX-" in stripped:
            last, closes = number, in_block
        elif closes and not in_block:
            last, closes = number, False
    if last is None:
        return 0
    end = last + 1
    while end < len(lines) and lines[end].strip() in ("", "//", "*", "*/"):
        end += 1
    return end


def _close(blocks: list[_Block], start: int | None, end: int, lines: Sequence[str], heading: str,
           is_heading: bool) -> None:
    """Append the block from `start` to `end` with its trailing blank lines dropped, when there is one."""
    if start is None:
        return
    while end >= start and not lines[end].strip():
        end -= 1
    if end >= start:
        blocks.append((start, end, heading, is_heading))


def _path(stack: Sequence[tuple[int, str]]) -> str:
    """Return a heading path: the titles from the outermost down, joined by ' > '."""
    return " > ".join(title for _, title in stack)


def _pack(blocks: Sequence[_Block], lines: Sequence[str], max_lines: int,
          max_chars: int) -> list[tuple[int, int, str]]:
    """Return (first line, last line, heading) per chunk: blocks packed in order, a new heading ending a chunk.

    A heading starts a new chunk unless the chunk so far holds only headings.
    A block over either limit is cut into pieces of whole lines.
    """
    chunks: list[tuple[int, int, str]] = []
    current: list[_Block] = []

    def size(first: int, last: int) -> tuple[int, int]:
        """Return (lines, characters) of lines `first` to `last`, each line counted with its newline."""
        return last - first + 1, sum(len(line) + 1 for line in lines[first:last + 1])

    def flush() -> None:
        """Emit the blocks gathered so far as one chunk, under the heading path of the last."""
        if current:
            chunks.append((current[0][0], current[-1][1], current[-1][2]))
            current.clear()

    for block in blocks:
        start, end, heading, is_heading = block
        if is_heading and current and not all(entry[3] for entry in current):
            flush()
        if current:
            count, chars = size(current[0][0], end)
            if count > max_lines or chars > max_chars:
                flush()
        count, chars = size(start, end)
        if count > max_lines or chars > max_chars:
            flush()
            chunks.extend((first, last, heading) for first, last in _pieces(lines, start, end, max_lines, max_chars))
            continue
        current.append(block)
    flush()
    return chunks


def _pieces(lines: Sequence[str], start: int, end: int, max_lines: int, max_chars: int) -> Iterable[tuple[int, int]]:
    """Yield (first, last) pieces of lines `start` to `end` within both limits; a piece never ends on a blank line."""
    first = start
    while first <= end:
        last, chars = first, len(lines[first]) + 1
        while last + 1 <= end and last + 1 - first + 1 <= max_lines and chars + len(lines[last + 1]) + 1 <= max_chars:
            last += 1
            chars += len(lines[last]) + 1
        trimmed = last
        while trimmed > first and not lines[trimmed].strip():
            trimmed -= 1
        yield first, trimmed
        first = last + 1
        while first <= end and not lines[first].strip():
            first += 1


def _windows(sequence: Sequence[str], size: int) -> list[tuple[str, ...]]:
    """Return every run of `size` consecutive items of `sequence`, in order."""
    return [tuple(sequence[offset:offset + size]) for offset in range(len(sequence) - size + 1)]


def _decoded(data: bytes) -> str | None:
    """Return `data` decoded as UTF-8, or None when it is not UTF-8."""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _sha256(data: bytes) -> str:
    """Return the sha256 hex digest of `data`."""
    return hashlib.sha256(data).hexdigest()


def _exact_keys(data: Any, keys: set[str], where: str) -> None:
    """Raise ValueError unless `data` is a mapping with exactly `keys`."""
    if not isinstance(data, dict) or set(data) != keys:
        found = sorted(data) if isinstance(data, dict) else type(data).__name__
        raise ValueError(f"{where} must hold exactly the keys {', '.join(sorted(keys))}, got {found}")


def _positive(value: Any, where: str) -> int:
    """Return `value` when it is an int of at least 1; else raise ValueError naming `where`."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{where} must be an int of at least 1, got {value!r}")
    return value


def _number(value: Any, where: str) -> float:
    """Return `value` as a float when it is a positive number; else raise ValueError naming `where`."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{where} must be a positive number, got {value!r}")
    return float(value)


def _fraction(value: Any, where: str) -> float:
    """Return `value` as a float when it is a number in [0, 1]; else raise ValueError naming `where`."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise ValueError(f"{where} must be a number in [0, 1], got {value!r}")
    return float(value)
