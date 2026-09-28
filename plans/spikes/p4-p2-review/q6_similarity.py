"""P2 review question 6 (task P4.15): what the notebook's token similarities give on known pairs.

Usage, with the interpreter of the repository's environment, from the root of
the tree under test (PYTHONPATH=<tree> pins the imports to it):

    <python> plans/spikes/p4-p2-review/q6_similarity.py (--upstream <checkout> | --bench-root <dir>)
        [--run <run dir>] [--tiktoken]

The 20 references are upstream LASSI's *_main sources: under
translated_code/input_codes/HeCBench of the pinned upstream checkout
(--upstream), or the byte-identical HeCBench files under a bench root that
tools/fetch_bench.py filled (--bench-root, laid out as
assets/bench/lassi-hecbench-10.yaml says). Each is read in text mode as
UTF-8, as the notebook reads it. For every reference, as the target:

- a perfect copy sent in a fenced block with each tag (cpp, c++, c, cuda,
  none) and read back with lassi.core.fragments.first_fence, the faithful
  port of the notebook's extraction; the line reports how many copies start
  with a line feed;
- the reference with one blank line added at the top, and in the middle
  (lines = text.splitlines(keepends=True), m = len(lines) // 2, candidate =
  lines[:m] + '\\n' + lines[m:], joined);
- the other-language original of the same app, unfenced and in a cpp fence.

With --run, it also reads each trial.json of a finished run and scores the
last attempt's target file against its reference, as the lassi profile does,
and again with a leading line feed removed. Each line gives min..max over its
pairs, at four decimals, of sim_t, sim_l, sim_t_c, the position-free
Python-tokenize ratio (difflib over (type, string) pairs, autojunk on, a
variant the notebook does not contain), and, with --tiktoken, sim_t_tiktoken
(cl100k_base read offline from TIKTOKEN_CACHE_DIR). It prints numbers and
counts only, never source or model text (OQ-018), and writes nothing.
"""

from __future__ import annotations

import argparse
import difflib
import json
import platform
from collections.abc import Callable, Sequence
from pathlib import Path

import yaml

from lassi.core.fragments import first_fence
from lassi.scoring import similarity

REPO = Path(__file__).resolve().parents[3]
MANIFEST = REPO / "assets" / "bench" / "lassi-hecbench-10.yaml"
UPSTREAM_PARTS = ("translated_code", "input_codes", "HeCBench")
EXTENSIONS = {"omp": "cpp", "cuda": "cu"}
OTHER = {"omp": "cuda", "cuda": "omp"}
TAGS = ("cpp", "c++", "c", "cuda", "")
Measure = Callable[[str, str], "float | None"]


def read_text(path: Path) -> str:
    """Return the file as text mode reads it: UTF-8, universal newlines."""
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def references(upstream: Path | None, bench_root: Path | None) -> dict[tuple[str, str], str]:
    """Return the 20 reference texts keyed by (app, language)."""
    items = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))["items"]
    found = {}
    for app, item in items.items():
        for lang in EXTENSIONS:
            if upstream is not None:
                path = upstream.joinpath(*UPSTREAM_PARTS) / app / f"{app}-{lang}_main.{EXTENSIONS[lang]}"
            else:
                spec = item["languages"][lang]
                path = bench_root / spec["dir"] / spec["files"][0]
            found[(app, lang)] = read_text(path)
    return found


def type_string(reference: str, candidate: str) -> float:
    """Return difflib's ratio over the (type, string) pairs of Python tokenize, autojunk on (position-free)."""
    tokens = similarity._python_tokens
    pairs = [[(token.type, token.string) for token in tokens(text)] for text in (reference, candidate)]
    return difflib.SequenceMatcher(None, pairs[0], pairs[1]).ratio()


def measures(with_tiktoken: bool) -> dict[str, Measure]:
    """Return the measures each line reports, by name."""
    found: dict[str, Measure] = {"sim_t": similarity.sim_t, "sim_l": similarity.sim_l, "sim_t_c": similarity.sim_t_c,
                                 "(type,string)": type_string}
    if with_tiktoken:
        found["sim_t_tiktoken"] = similarity.sim_t_tiktoken
    return found


def span(values: Sequence[float | None]) -> str:
    """Return 'min..max' at four decimals, with the count of None values when there are any."""
    numbers = [value for value in values if value is not None]
    text = f"{min(numbers):.4f}..{max(numbers):.4f}" if numbers else "-"
    missing = len(values) - len(numbers)
    return text + (f" ({missing} None)" if missing else "")


def report(label: str, pairs: Sequence[tuple[str, str]], named: dict[str, Measure]) -> None:
    """Print one line: the label, the pair count, and each measure's range over the pairs."""
    cells = [f"{name} {span([measure(ref, cand) for ref, cand in pairs])}" for name, measure in named.items()]
    print(f"{label} [{len(pairs)} pairs]: " + "; ".join(cells))


def middle_blank(text: str) -> str:
    """Return `text` with one blank line inserted after its first half of lines."""
    lines = text.splitlines(keepends=True)
    half = len(lines) // 2
    return "".join(lines[:half]) + "\n" + "".join(lines[half:])


def run_pairs(run: Path, refs: dict[tuple[str, str], str]) -> tuple[list[tuple[str, str]], int]:
    """Return (reference, last attempt's target file) per trial of `run` and how many files start with a line feed."""
    pairs, leading = [], 0
    for path in sorted(run.rglob("trial.json")):
        trial = json.loads(path.read_text(encoding="utf-8"))
        if not trial["attempts"]:
            continue
        bench = trial["bench_item"]
        target = bench["direction"].split("-")[1]
        name = f"main.{EXTENSIONS[target]}"
        text = trial["attempts"][-1]["files"].get(name, "")
        leading += text.startswith("\n")
        pairs.append((refs[(bench["item"], target)], text))
    return pairs, leading


def main() -> None:
    """Print the lines the module docstring lists."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument("--upstream", type=Path)
    where.add_argument("--bench-root", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--tiktoken", action="store_true")
    args = parser.parse_args()
    refs = references(args.upstream, args.bench_root)
    named = measures(args.tiktoken)
    version = f"; tiktoken {similarity.tiktoken_version()}" if args.tiktoken else ""
    print(f"python {platform.python_version()}{version}; {len(refs)} references")
    for tag in TAGS:
        copies = [(text, first_fence(f"```{tag}\n{text}```\n").text) for text in refs.values()]
        leading = sum(candidate.startswith("\n") for _, candidate in copies)
        report(f"perfect copy, fence tag {tag!r}, leading line feed {leading}/{len(copies)}", copies, named)
    report("blank line added at the top", [(text, "\n" + text) for text in refs.values()], named)
    report("blank line added in the middle", [(text, middle_blank(text)) for text in refs.values()], named)
    siblings = [(text, refs[(app, OTHER[lang])]) for (app, lang), text in refs.items()]
    report("other-language original, unfenced", siblings, named)
    fenced = [(ref, first_fence(f"```cpp\n{cand}```\n").text) for ref, cand in siblings]
    report("other-language original, cpp fence", fenced, named)
    if args.run is not None:
        pairs, leading = run_pairs(args.run, refs)
        report(f"run {args.run.name}, last target file, leading line feed {leading}/{len(pairs)}", pairs, named)
        stripped = [(ref, cand[1:] if cand.startswith("\n") else cand) for ref, cand in pairs]
        report(f"run {args.run.name}, leading line feed removed", stripped, named)


if __name__ == "__main__":
    main()
