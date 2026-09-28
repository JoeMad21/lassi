"""P2 review question 1 (task P4.15): which faithful recipes the loader accepts.

Usage, with the interpreter of the repository's environment:

    PYTHONPATH=<tree> <python> plans/spikes/p4-p2-review/q1_faithful_load.py <tree>

<tree> is an export of the commit under test (for example `git archive 1ccf1fb`
unpacked into a scratch directory), so the imports come from that commit and
not from a working tree mid-edit. The script writes four child recipes into
<tree>/projects/lassi-demo/ (a scratch copy, never the repository) and loads
each, plus tests/fixtures/recipes/p1-dry-run.yaml and projects/lassi-repro,
with the default registry. It prints one line per recipe: LOADED with the
resolved faithful flag, executor, toolchains, and correction cap, or REFUSED
with the loader's message. It starts no process and opens no socket.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

CHILDREN = {
    "A-faithful-cpu-proxy": "extends: rngd-cpu.yaml\nfaithful: true\n",
    "B-faithful-gcc-native": "extends: rngd-cpu.yaml\nfaithful: true\ntoolchain: {omp: gcc-native}\n",
    "C-faithful-nvcpp-native": "extends: rngd-cpu.yaml\nfaithful: true\ntoolchain: {omp: nvcpp-cc80}\n",
}
COMPONENT_MODULES = ("lassi.cli", "lassi.llm", "lassi.toolchains", "lassi.oracles", "lassi.executors")


def describe(recipe: object) -> str:
    """Return the resolved keys the question turns on."""
    data = recipe.data  # type: ignore[attr-defined]
    loop = data.get("loop", {})
    return (f"faithful: {data.get('faithful')} executor: {data.get('executor')} toolchain: {data.get('toolchain')} "
            f"max_corrections: {loop.get('max_corrections')} chain: {tuple(recipe.chain)}")  # type: ignore[attr-defined]


def main() -> None:
    """Load each recipe and print the outcome."""
    tree = Path(sys.argv[1]).resolve()
    for name in COMPONENT_MODULES:
        importlib.import_module(name)
    import lassi
    from lassi.core.recipe import RecipeError, load_recipe

    print("lassi imported from", Path(lassi.__file__).resolve().relative_to(tree).as_posix())
    roots = [tree / "projects"]
    paths = {}
    for name, text in CHILDREN.items():
        path = tree / "projects" / "lassi-demo" / f"{name}.yaml"
        path.write_text(text, encoding="utf-8")
        paths[name] = path
    paths["D-p1-dry-run"] = tree / "tests" / "fixtures" / "recipes" / "p1-dry-run.yaml"
    paths["E-lassi-repro"] = tree / "projects" / "lassi-repro" / "recipe.yaml"
    for name, path in paths.items():
        try:
            print(name, "LOADED", describe(load_recipe(path, roots=roots)))
        except RecipeError as error:
            print(name, "REFUSED", str(error).replace(str(tree), "<tree>"))


if __name__ == "__main__":
    main()
