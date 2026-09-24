"""Generate the lassi-2024 prompt set and its two context packs from pinned upstream LASSI (task P1.3).

Usage:

    uv run tools/extract_lassi_assets.py [--upstream DIR] [--out DIR]

`--upstream` (default third_party/LASSI) must be a git checkout of its own
whose HEAD is upstream LASSI's pinned commit
74b46812523f2ff79b53b6880a4521690d7478b0 (bible, Source Papers, LASSI).
Anything else is refused with exit status 1, a message naming the pin, and no
file written. The two upstream files are read from the pinned commit's
objects with `git show`, so local edits in the checkout change nothing.

The tool never imports or runs upstream code: `prompt_dictionary.py` and each
notebook cell are parsed with `ast`, the notebook itself with `json`, and the
only process started is git.

It writes three trees under `--out` (default assets/), each holding
MANIFEST.yaml and one `<key>.txt` per manifest entry (a fresh run writes
nothing else; a file an earlier run left that the new manifest does not list
stays in place, and the loader ignores it):

- `prompts/lassi-2024/`: every value of every dictionary in
  `prompt_dictionary.py` except the context knowledge one, keyed
  `<dictionary>.<entry>`, and the notebook string literals the faithful
  stages use, keyed by stage role (NOTEBOOK_KEYS).
- `context/openmp-4.0-card/` and `context/cuda-12.5-ch5/`: one entry each,
  the context knowledge value for that language (PACKS).

A fragment file holds one upstream string value verbatim: UTF-8, no newline
translation, nothing filled in or added. The stages join fragments in
upstream's order; joiners that are only whitespace stay with the stages.
MANIFEST.yaml (plain ASCII, LF) lists each entry's key, source file, notebook
cell id (notebook entries only), and the sha256 of the file's bytes.

Only the manifests are tracked; the fragment files are gitignored (OQ-018).
Rerunning writes the same bytes. Exit status: 0 on success, 1 on a refusal
(wrong checkout, or an upstream layout this tool does not expect).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
DEFAULT_UPSTREAM = REPO / "third_party" / "LASSI"
DEFAULT_OUT = REPO / "assets"
# Upstream LASSI's pinned commit (bible, Source Papers, LASSI). The checkout must be exactly here.
PIN = "74b46812523f2ff79b53b6880a4521690d7478b0"
DICTIONARY_FILE = "prompt_dictionary.py"
NOTEBOOK_FILE = "LASSI_pipeline_v0.ipynb"
MANIFEST = "MANIFEST.yaml"
PROMPT_TREE = "prompts/lassi-2024"
# The dictionary that holds the context knowledge, and the entry each pack tree holds.
PACK_DICTIONARY = "codeknowledge_dict"
PACKS = {"context/openmp-4.0-card": "omp", "context/cuda-12.5-ch5": "cuda"}
# A key is one plain path segment; its file is <key>.txt.
KEY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
# Environment variables that would point git at a repository other than the checkout.
REPO_VARS = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
             "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_COMMON_DIR", "GIT_NAMESPACE")

# Stage-facing keys for the notebook literals, per (notebook function, assigned name), in source order. Each
# tuple lists one key per non-blank string literal that upstream joins with `+` into that name; a different
# count at the pin is refused. The comment after a key says where upstream uses the literal.
NOTEBOOK_KEYS: dict[tuple[str, str], tuple[str, ...]] = {
    ("code_knowledge_llm", "content_prompt"): (
        "summarize_context.intro",  # opens the context summary request
        "summarize_context.omp_note",  # appended only when the target is OpenMP
        "summarize_context.outro",  # closes the request; the context text follows
    ),
    ("code_description_llm", "content_prompt"): (
        "describe_source.intro",  # the source description request; the source text follows
    ),
    ("auto_code_llmgeneration_pipeline", "content"): (
        "generate.context_open",  # before the context text
        "generate.context_close",  # after the context text
        "generate.summary_open",  # before the context summary
        "generate.summary_close",  # after the context summary
        "generate.context_lead",  # after the summary block
        "generate.description_lead",  # before the source description
        "generate.no_context_lead",  # the branch without context, after the general system prompt
        "generate.request_lead",  # before the direction's translation prompt
    ),
    ("auto_code_llmgeneration_pipeline", "error_correction_prompt_intro"): (
        "correct.run_error_head",  # run error branch, before the compiler command
        "correct.run_error_tail",  # run error branch, after the flags
        "correct.compile_error_head",  # compile error branch, before the compiler command
        "correct.compile_error_tail",  # compile error branch, after the flags
    ),
    ("auto_code_llmgeneration_pipeline", "error_correction_prompt_outro"): (
        "correct.outro",  # after the error text
    ),
    ("execute_code", "return_result"): (
        "execute.ok",  # the report of a clean run, which no prompt carries
        "execute.exit_lead",  # a failed run's report (the run error text), before the return code
        "execute.segfault",  # appended when the return code is -11
        "execute.exception_lead",  # before an exception raised while reading the run's output
        "execute.stderr_lead",  # before the run's stderr, when it wrote any
    ),
    ("experimental_setup", "code_compiler"): (
        "setup.cuda.compiler",  # CUDA target branch
        "setup.omp.compiler",  # OpenMP target branch
    ),
    ("experimental_setup", "code_compiler_kwds"): (
        "setup.cuda.flags",  # CUDA target branch
        "setup.omp.flags",  # OpenMP target branch
    ),
}

HEADER = (
    f"# Generated by tools/extract_lassi_assets.py from upstream LASSI at {PIN}.\n"
    "# Only this manifest is tracked; the fragment files beside it are generated and gitignored (OQ-018).\n"
)


class Refusal(RuntimeError):
    """The checkout is not upstream LASSI at the pin, or its files do not have the layout this tool reads."""


@dataclass(frozen=True)
class Fragment:
    """One upstream string value and where it came from."""

    key: str
    source: str
    cell: str | None
    text: str


def _git(checkout: Path, *args: str) -> subprocess.CompletedProcess:
    """Run git with `args` in `checkout`, with every variable that redirects git removed; return the process."""
    env = {name: value for name, value in os.environ.items() if name not in REPO_VARS}
    try:
        return subprocess.run(["git", "-C", str(checkout), *args], capture_output=True, env=env, check=False)
    except OSError as error:
        raise Refusal(f"cannot run git in {checkout} to check it is upstream LASSI at {PIN}: {error}") from error


def check_pin(checkout: Path) -> None:
    """Raise Refusal unless `checkout` is its own git checkout (not a directory inside one) with HEAD at PIN."""
    if not checkout.is_dir():
        raise Refusal(f"{checkout} is not a directory; expected a git checkout of upstream LASSI at {PIN}")
    done = _git(checkout, "rev-parse", "--show-toplevel", "HEAD")
    lines = done.stdout.decode("utf-8", "replace").splitlines()
    if done.returncode != 0 or len(lines) != 2:
        raise Refusal(f"{checkout} is not a git checkout; expected upstream LASSI at {PIN}")
    toplevel, head = lines
    try:
        own = os.path.samefile(toplevel, checkout)
    except OSError:
        own = False
    if not own:
        raise Refusal(f"{checkout} is not a checkout of its own (it lies inside {toplevel}); expected {PIN}")
    if head != PIN:
        raise Refusal(f"{checkout} is at {head}, not upstream LASSI's pinned commit {PIN}")


def read_pinned(checkout: Path, name: str) -> str:
    """Return the file `name` as stored in the pinned commit, decoded as UTF-8 with no newline translation."""
    done = _git(checkout, "show", f"{PIN}:{name}")
    if done.returncode != 0:
        raise Refusal(f"commit {PIN} in {checkout} has no readable {name}")
    try:
        return done.stdout.decode("utf-8")
    except UnicodeDecodeError as error:
        raise Refusal(f"{name} at {PIN} is not UTF-8: {error}") from error


def parse_dictionaries(source: str) -> dict[str, dict[str, str]]:
    """Return {name: {entry: value}} for every dict literal of string constants assigned in prompt_dictionary.py.

    A dict is named by its assignment target (an attribute or a plain name).
    Dicts come in source order, entries in literal order.
    """
    found: dict[str, dict[str, str]] = {}
    assigns = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Assign)
               and isinstance(node.value, ast.Dict)]
    for node in sorted(assigns, key=lambda node: (node.lineno, node.col_offset)):
        target = node.targets[0]
        name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", "")
        if not name or name in found:
            raise Refusal(f"{DICTIONARY_FILE} at {PIN}: a dict has no name or a repeated name ({name!r})")
        values: dict[str, str] = {}
        for key, value in zip(node.value.keys, node.value.values, strict=True):
            plain = all(isinstance(part, ast.Constant) and isinstance(part.value, str) for part in (key, value))
            if not plain:
                raise Refusal(f"{DICTIONARY_FILE} at {PIN}: {name} holds an entry that is not a string constant")
            values[key.value] = value.value
        found[name] = values
    return found


def function_cells(notebook: str) -> dict[str, tuple[str, ast.FunctionDef]]:
    """Return {function name: (cell id, its definition)} for every function defined at the top of a code cell.

    Cells that do not parse as Python (notebook magics) are skipped. A
    function defined in two cells is refused.
    """
    cells: dict[str, tuple[str, ast.FunctionDef]] = {}
    for cell in json.loads(notebook).get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        try:
            tree = ast.parse(source if isinstance(source, str) else "".join(source))
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                if node.name in cells:
                    raise Refusal(f"{NOTEBOOK_FILE} at {PIN}: two cells define {node.name}")
                cells[node.name] = (str(cell.get("id", "")), node)
    return cells


def _operands(expr: ast.expr) -> list[ast.expr]:
    """Flatten a chain of `+` into its operands, left to right."""
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
        return _operands(expr.left) + _operands(expr.right)
    return [expr]


def assigned_literals(function: ast.FunctionDef, name: str) -> list[str]:
    """Return the non-blank string literals that are direct `+` operands of assignments to `name`, in source order.

    Only `function` is walked, never the rest of its cell.
    """
    statements = []
    for node in ast.walk(function):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AugAssign):
            targets = [node.target]
        else:
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            statements.append(node)
    literals = []
    for node in sorted(statements, key=lambda node: (node.lineno, node.col_offset)):
        literals.extend(operand.value for operand in _operands(node.value)
                        if isinstance(operand, ast.Constant) and isinstance(operand.value, str)
                        and operand.value.strip())
    return literals


def notebook_fragments(notebook: str) -> list[Fragment]:
    """Return one Fragment per literal NOTEBOOK_KEYS names, refusing a count that differs from the table."""
    cells = function_cells(notebook)
    fragments = []
    for (function, name), keys in NOTEBOOK_KEYS.items():
        if function not in cells or not cells[function][0]:
            raise Refusal(f"{NOTEBOOK_FILE} at {PIN}: no cell with an id defines {function}")
        cell_id, definition = cells[function]
        literals = assigned_literals(definition, name)
        if len(literals) != len(keys):
            raise Refusal(f"{NOTEBOOK_FILE} at {PIN}: {function} joins {len(literals)} literals into {name}, "
                          f"expected {len(keys)}")
        fragments.extend(Fragment(key, NOTEBOOK_FILE, cell_id, text) for key, text in zip(keys, literals, strict=True))
    return fragments


def build_trees(dictionary: str, notebook: str) -> dict[str, list[Fragment]]:
    """Return {tree path under the assets root: its fragments} for the prompt set and both packs."""
    dictionaries = parse_dictionaries(dictionary)
    packs = dictionaries.get(PACK_DICTIONARY, {})
    trees: dict[str, list[Fragment]] = {PROMPT_TREE: [
        Fragment(f"{name}.{entry}", DICTIONARY_FILE, None, text)
        for name, values in dictionaries.items() if name != PACK_DICTIONARY for entry, text in values.items()
    ]}
    trees[PROMPT_TREE].extend(notebook_fragments(notebook))
    for tree, entry in PACKS.items():
        if entry not in packs:
            raise Refusal(f"{DICTIONARY_FILE} at {PIN}: {PACK_DICTIONARY} has no entry {entry!r}")
        trees[tree] = [Fragment(f"{PACK_DICTIONARY}.{entry}", DICTIONARY_FILE, None, packs[entry])]
    for tree, fragments in trees.items():
        keys = [fragment.key for fragment in fragments]
        bad = [key for key in keys if not KEY_RE.fullmatch(key)]
        if bad or len(keys) != len(set(keys)):
            raise Refusal(f"{tree}: keys at {PIN} are not unique plain path segments: {bad or 'repeated keys'}")
    return trees


def manifest_text(fragments: list[Fragment]) -> str:
    """Return the MANIFEST.yaml text for `fragments`: the header, then key, source, cell, and sha256 per entry."""
    entries = []
    for fragment in fragments:
        entry: dict[str, str] = {"key": fragment.key, "source": fragment.source}
        if fragment.cell is not None:
            entry["cell"] = fragment.cell
        entry["sha256"] = hashlib.sha256(fragment.text.encode("utf-8")).hexdigest()
        entries.append(entry)
    return HEADER + yaml.safe_dump({"entries": entries}, sort_keys=False, default_flow_style=False)


def _write_if_changed(path: Path, data: bytes) -> None:
    """Write `data` to `path` unless the file already holds exactly those bytes."""
    if not (path.is_file() and path.read_bytes() == data):
        path.write_bytes(data)


def write_tree(directory: Path, fragments: list[Fragment]) -> None:
    """Write one `<key>.txt` per fragment and the tree's MANIFEST.yaml into `directory`."""
    directory.mkdir(parents=True, exist_ok=True)
    for fragment in fragments:
        _write_if_changed(directory / f"{fragment.key}.txt", fragment.text.encode("utf-8"))
    _write_if_changed(directory / MANIFEST, manifest_text(fragments).encode("ascii"))


def main(argv: list[str] | None = None) -> int:
    """Check the checkout, parse the pinned upstream files, and write the three trees; return the exit status."""
    parser = argparse.ArgumentParser(description="Generate the lassi-2024 prompt set and context packs.")
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM, help="upstream LASSI checkout at the pin")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="assets root to write the trees under")
    args = parser.parse_args(argv)
    try:
        check_pin(args.upstream)
        trees = build_trees(read_pinned(args.upstream, DICTIONARY_FILE), read_pinned(args.upstream, NOTEBOOK_FILE))
    except Refusal as refusal:
        print(f"extract_lassi_assets: refused: {refusal}", file=sys.stderr)
        return 1
    for tree, fragments in trees.items():
        write_tree(args.out / tree, fragments)
        print(f"{(args.out / tree).as_posix()}: {len(fragments)} entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
