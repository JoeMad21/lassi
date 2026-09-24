"""Tests for tools/extract_lassi_assets.py and the generated lassi-2024 trees (task P1.3).

Bible: Source Papers (LASSI pipeline steps 2 to 4), Design Principles 3 and 4,
Readability Standards (Prompts row), Repository Layout (`assets/prompts`,
`assets/context`). OQ-018 is binding: upstream text is generated into
gitignored trees and only the manifests are committed.

The contract these tests fix:

- `main(argv)` takes `--upstream <checkout>` (default `third_party/LASSI`)
  and `--out <assets root>` (default `assets/`). It refuses, with a nonzero
  status and a message naming 74b4681, any directory that is not a git
  checkout at 74b46812523f2ff79b53b6880a4521690d7478b0, and then writes
  nothing.
- It parses `prompt_dictionary.py` and the notebook cells with `ast` and
  `json`. It imports nothing from upstream and starts no process but git.
- It writes three trees under the assets root: `prompts/lassi-2024/`,
  `context/openmp-4.0-card/`, and `context/cuda-12.5-ch5/`. A fresh
  extraction leaves in each `MANIFEST.yaml` and one `<key>.txt` per manifest
  entry, and nothing else. The manifest fields are listed in conftest.py
  (Manifests.entries).
- Each fragment file holds, verbatim (UTF-8, no newline translation), the
  upstream string value its key names (conftest.py, Upstream.value_of): the
  dictionary value `<dictionary>.<entry>`, or the notebook literal at the
  source position NOTEBOOK_KEYS gives the key, in the cell its manifest entry
  names. The manifest lists the notebook keys in NOTEBOOK_KEYS order. The
  stages join fragments in upstream's order; whitespace-only joiners stay
  with the stages.
- The prompt set holds every value of every dictionary except the context
  knowledge one, and every literal the stages use (see conftest.py). Each
  pack holds exactly one entry: the context knowledge value for its
  language.
- Rerunning changes nothing, and a fresh extraction reproduces the committed
  manifests byte for byte.
- `.gitignore` keeps every generated file out of git and every manifest in.
  No tracked file outside `third_party/` holds a generated fragment of 40 or
  more characters. The only exemption: a compiler or flag literal of
  `experimental_setup` that docs/BIBLE.md already records (Source Papers,
  LASSI, compile flags), found when the test runs. Every other setup
  literal, upstream's compiler path included, is guarded like the rest.

Assertion messages name keys, cells, and counts, never upstream text, so a
failure prints none of it. No value here is a measurement.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import Manifests, Upstream

REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "tools" / "extract_lassi_assets.py"
UPSTREAM_PIN = "74b46812523f2ff79b53b6880a4521690d7478b0"
DICTIONARY_FILE = "prompt_dictionary.py"
NOTEBOOK_FILE = "LASSI_pipeline_v0.ipynb"
PROMPT_TREE = "prompts/lassi-2024"
PACKS = {
    "openmp-4.0-card": ("codeknowledge_dict", "omp"),
    "cuda-12.5-ch5": ("codeknowledge_dict", "cuda"),
}
TREES = (PROMPT_TREE, *(f"context/{name}" for name in PACKS))
MANIFEST = "MANIFEST.yaml"
LEAK_MIN = 40
BIBLE = REPO / "docs" / "BIBLE.md"

# What the extractor must never do to read upstream: import it or execute it.
FORBIDDEN_MODULES = frozenset({"prompt_dictionary", "importlib", "runpy"})
FORBIDDEN_CALLS = frozenset({"exec", "eval", "compile", "__import__"})


def git(*args: str, cwd: Path = REPO, check: bool = True, env: dict | None = None) -> subprocess.CompletedProcess:
    """Run git with `args` in `cwd` and return the finished process (text mode)."""
    return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, check=check, env=env)


def files_under(root: Path) -> dict[str, bytes]:
    """Return {path relative to root, with forward slashes: bytes} for every file under `root`."""
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


def fragments(root: Path, tree: str, manifests: Manifests) -> list[tuple[dict, bytes]]:
    """Return (manifest entry, file bytes) for every entry of the tree `root/tree`."""
    base = root / tree
    return [(entry, (base / f"{entry['key']}.txt").read_bytes()) for entry in manifests.entries(base)]


# ---------------------------------------------------------------- what it writes


def test_extractor_writes_the_prompt_set_and_both_packs(extraction, manifests, upstream) -> None:
    root = extraction.root()
    expected: set[str] = set()
    for tree in TREES:
        entries = manifests.entries(root / tree)
        names = {MANIFEST, *(f"{entry['key']}.txt" for entry in entries)}
        present = {path.name for path in (root / tree).iterdir()}
        extra, lacking = sorted(present - names), sorted(names - present)
        assert not extra and not lacking, f"{tree} holds {extra} beyond its manifest and lacks {lacking}"
        for entry in entries:
            data = (root / tree / f"{entry['key']}.txt").read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            assert digest == entry["sha256"], f"{tree}/{entry['key']}: sha256 differs"
            if entry["source"] == NOTEBOOK_FILE:
                known = entry["cell"] in upstream.cells
                assert known, f"{tree}/{entry['key']}: no notebook cell {entry['cell']!r}"
        expected |= {f"{tree}/{name}" for name in names}
    stray = sorted(set(files_under(root)) - expected)
    assert not stray, f"the extractor wrote files outside the three trees: {stray}"


def test_every_fragment_is_the_value_its_key_names_byte_for_byte(extraction, manifests, upstream) -> None:
    root = extraction.root()
    for tree in TREES:
        for entry, data in fragments(root, tree, manifests):
            same = data == upstream.value_of(entry).encode("utf-8")
            assert same, f"{tree}/{entry['key']} is not byte for byte the upstream value its key names"


def test_notebook_keys_follow_the_source_order_of_their_literals(extraction, manifests, upstream) -> None:
    entries = manifests.entries(extraction.root() / PROMPT_TREE)
    listed = [entry["key"] for entry in entries if entry["source"] == NOTEBOOK_FILE]
    expected = list(upstream.notebook_values())
    assert listed == expected, f"{PROMPT_TREE} lists notebook keys {listed}, expected {expected}"


def test_prompt_set_covers_the_dictionary_and_the_literals_the_stages_use(extraction, manifests, upstream) -> None:
    pairs = fragments(extraction.root(), PROMPT_TREE, manifests)
    pack_dictionaries = {dictionary for dictionary, _ in PACKS.values()}
    wanted = {f"{name}.{item}" for name, values in upstream.dictionaries.items() if name not in pack_dictionaries
              for item in values}
    listed = {entry["key"] for entry, _ in pairs if entry["source"] == DICTIONARY_FILE}
    assert listed == wanted, f"{PROMPT_TREE} lacks {sorted(wanted - listed)} and adds {sorted(listed - wanted)}"
    for cell_id, literals in upstream.required().items():
        found = {data.decode("utf-8") for entry, data in pairs if entry.get("cell") == cell_id}
        missing = len(literals - found)
        assert missing == 0, f"{missing} literals of notebook cell {cell_id} the stages use have no fragment"


@pytest.mark.parametrize("pack", sorted(PACKS))
def test_each_pack_holds_its_context_knowledge_value(pack: str, extraction, manifests, upstream) -> None:
    pairs = fragments(extraction.root(), f"context/{pack}", manifests)
    assert len(pairs) == 1, f"context/{pack} should hold one entry, holds {len(pairs)}"
    entry, data = pairs[0]
    dictionary, key = PACKS[pack]
    assert entry["source"] == DICTIONARY_FILE and entry["key"] == f"{dictionary}.{key}"
    same = data == upstream.dictionaries[dictionary][key].encode("utf-8")
    assert same, f"context/{pack}/{entry['key']}.txt differs from {dictionary}[{key!r}]"


def test_literals_come_from_the_named_function_not_the_rest_of_its_cell(tool_module: ModuleType) -> None:
    # Synthetic cell text written here: two functions in one cell assign the same name.
    source = 'def wanted(value):\n    text = "kept" + value\n\n\ndef other():\n    text = "not kept"\n'
    notebook = json.dumps({"cells": [{"cell_type": "code", "id": "cell-1", "source": source}]})
    cell_id, node = tool_module.function_cells(notebook)["wanted"]
    assert cell_id == "cell-1"
    assert tool_module.assigned_literals(node, "text") == ["kept"]


def test_rerun_changes_nothing(extractor, upstream, tmp_path: Path) -> None:
    out = tmp_path / "assets"
    first = extractor.run(upstream.path, out)
    assert first[0] == 0, f"the first run exited {first[0]}: {first[1][-2000:]}"
    before = files_under(out)
    second = extractor.run(upstream.path, out)
    assert second[0] == 0, f"the rerun exited {second[0]}: {second[1][-2000:]}"
    after = files_under(out)
    assert sorted(after) == sorted(before), "the rerun added or removed files"
    changed = sorted(name for name in before if after[name] != before[name])
    assert not changed, f"the rerun changed {changed}"


def test_committed_manifests_match_a_fresh_extraction(extraction) -> None:
    root = extraction.root()
    for tree in TREES:
        committed = REPO / "assets" / tree / MANIFEST
        assert committed.is_file(), f"assets/{tree}/{MANIFEST} is not in the repository"
        same = committed.read_bytes() == (root / tree / MANIFEST).read_bytes()
        assert same, f"assets/{tree}/{MANIFEST} differs from what the extractor writes at the pin"


# ---------------------------------------------------------------- refusals and guards


def move_off_the_pin(clone: Path) -> None:
    """Detach `clone` at a commit other than the pin: its parent, or a new commit with the pin's tree."""
    parent = git("rev-parse", "--verify", "--quiet", f"{UPSTREAM_PIN}~1^{{commit}}", cwd=clone, check=False)
    if parent.returncode == 0:
        target = parent.stdout.strip()
    else:
        who = {"GIT_AUTHOR_NAME": "fixture", "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
               "GIT_COMMITTER_NAME": "fixture", "GIT_COMMITTER_EMAIL": "fixture@example.invalid"}
        made = git("commit-tree", f"{UPSTREAM_PIN}^{{tree}}", "-p", UPSTREAM_PIN, "-m", "fixture: off the pin",
                   cwd=clone, env={**os.environ, **who})
        target = made.stdout.strip()
    git("checkout", "--quiet", "--detach", target, cwd=clone)
    assert git("rev-parse", "HEAD", cwd=clone).stdout.strip() != UPSTREAM_PIN


def test_extractor_refuses_a_checkout_off_the_pin(extractor, upstream, tmp_path: Path) -> None:
    clone = tmp_path / "LASSI"
    git("clone", "--quiet", "--no-checkout", str(upstream.path), str(clone), cwd=tmp_path)
    move_off_the_pin(clone)
    out = tmp_path / "assets"
    status, output = extractor.run(clone, out)
    assert status != 0, "the extractor accepted a checkout that is not at 74b4681"
    named = "74b4681" in output
    assert named, "the refusal names the pinned commit"
    assert not (out.exists() and files_under(out)), "a refused run wrote files"


def test_extractor_refuses_a_directory_that_is_no_checkout(extractor, upstream, tmp_path: Path) -> None:
    plain = tmp_path / "LASSI"
    plain.mkdir()
    for name in (DICTIONARY_FILE, NOTEBOOK_FILE):
        shutil.copyfile(upstream.path / name, plain / name)
    out = tmp_path / "assets"
    status, output = extractor.run(plain, out)
    assert status != 0, "the extractor accepted a directory that is not a git checkout"
    named = "74b4681" in output
    assert named, "the refusal names the pinned commit"
    assert not (out.exists() and files_under(out)), "a refused run wrote files"


def test_extractor_source_never_imports_or_executes_upstream() -> None:
    assert TOOL.is_file(), "tools/extract_lassi_assets.py is missing (task P1.3 writes it)"
    tree = ast.parse(TOOL.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module or ""]
        else:
            modules = []
        bad = [name for name in modules if name.split(".")[0] in FORBIDDEN_MODULES]
        assert not bad, f"line {node.lineno}: the extractor imports {bad}"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in FORBIDDEN_CALLS, f"line {node.lineno}: the extractor calls {node.func.id}"


class GitOnly(subprocess.Popen):
    """A Popen that lets only git start, so the extractor cannot run upstream code in another process."""

    def __init__(self, args, *rest, **kwargs) -> None:
        program = args[0] if isinstance(args, (list, tuple)) else str(args).split()[0]
        assert Path(str(program)).stem.lower() == "git", f"the extractor started {program!r}; only git may run"
        super().__init__(args, *rest, **kwargs)


def test_extractor_run_loads_no_upstream_module(extractor, upstream, tmp_path: Path, monkeypatch) -> None:
    def no_shell(command: str) -> int:
        raise AssertionError("the extractor called os.system")

    monkeypatch.setattr(subprocess, "Popen", GitOnly)
    monkeypatch.setattr(os, "system", no_shell)
    path_before = list(sys.path)
    status, output = extractor.run(upstream.path, tmp_path / "assets")
    assert status == 0, f"the extractor exited {status}: {output[-2000:]}"
    root = upstream.path.resolve()
    loaded = sorted(name for name, module in list(sys.modules.items())
                    if getattr(module, "__file__", None) and root in Path(module.__file__).resolve().parents)
    assert not loaded, f"the extractor imported upstream modules: {loaded}"
    imported = "prompt_dictionary" in sys.modules
    assert not imported, "the extractor imported prompt_dictionary"
    added = [entry for entry in sys.path if entry not in path_before
             and root in (Path(entry).resolve(), *Path(entry).resolve().parents)]
    assert not added, f"the extractor put upstream on sys.path: {added}"


# ---------------------------------------------------------------- what git keeps


def ignored(path: str) -> bool:
    """Return True when .gitignore ignores `path` (checked against the patterns, tracked or not)."""
    done = git("check-ignore", "--quiet", "--no-index", path, check=False)
    assert done.returncode in (0, 1), f"git check-ignore failed: {done.stderr.strip()}"
    return done.returncode == 0


@pytest.mark.parametrize("tree", TREES)
def test_gitignore_keeps_generated_text_out_and_the_manifest_in(tree: str) -> None:
    for name in ("fragment.txt", "extra.yaml", "nested/fragment.txt"):
        assert ignored(f"assets/{tree}/{name}"), f"assets/{tree}/{name} is not ignored"
    assert not ignored(f"assets/{tree}/{MANIFEST}"), f"assets/{tree}/{MANIFEST} is ignored"


def test_gitignore_leaves_the_hand_written_prompt_set_tracked() -> None:
    assert not ignored("assets/prompts/p0-smoke/generate.txt")


def test_only_the_manifests_are_tracked_in_the_generated_trees() -> None:
    listed = git("ls-files", "--", "assets/prompts/lassi-2024", "assets/context").stdout.split()
    assert sorted(listed) == sorted(f"assets/{tree}/{MANIFEST}" for tree in TREES)


def test_committed_manifests_have_the_manifest_fields(manifests) -> None:
    for tree in TREES:
        manifests.entries(REPO / "assets" / tree)


def generated_texts(root: Path, manifests: Manifests, upstream: Upstream) -> dict[str, str]:
    """Return {label: text} for every generated fragment and oracle value of LEAK_MIN or more characters.

    Left out: the compiler and flag literals of experimental_setup that docs/BIBLE.md records now.
    """
    bible = BIBLE.read_bytes().decode("utf-8").replace("\r\n", "\n")
    exempt = {text for text in upstream.setup_literals() if text in bible}
    texts: dict[str, str] = {}
    for tree in TREES:
        for entry, data in fragments(root, tree, manifests):
            texts[f"{tree}/{entry['key']}"] = data.decode("utf-8")
    oracle = upstream.dictionary_values(packs=False) | upstream.dictionary_values(packs=True)
    for literals in upstream.required().values():
        oracle |= literals
    for number, text in enumerate(sorted(oracle - set(texts.values()))):
        texts[f"upstream value {number}"] = text
    return {label: text for label, text in texts.items() if len(text) >= LEAK_MIN and text not in exempt}


def test_no_tracked_file_holds_a_generated_fragment(extraction, manifests, upstream) -> None:
    texts = generated_texts(extraction.root(), manifests, upstream)
    assert texts, "no generated fragment reaches the length the guard checks"
    tracked = git("ls-files", "-z").stdout.split("\0")
    hits: list[str] = []
    for name in tracked:
        path = REPO / name
        if not name or name.startswith("third_party/") or not path.is_file():
            continue
        body = path.read_bytes().decode("latin-1").replace("\r\n", "\n")
        hits.extend(f"{name} holds {label}" for label, text in texts.items() if text in body)
    assert not hits, f"tracked files hold generated upstream text (OQ-018): {hits}"
