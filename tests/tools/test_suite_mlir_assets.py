"""Tests for the suite job's TTIR and D2M asset steps (task P17.47; plans/runs/suite).

The TTIR and D2M API references (ttir-ref-5f396fd6, d2m-ref-5f396fd6) and
tests indexes (tt-mlir-ttir-tests, tt-mlir-d2m-tests) are built at job time,
as the ttkernel ones are, only when a selected recipe needs one: recipes.py
needs maps a recipe's context and retrieval sections to context:<pack> and
retrieval:<index>, run-suite.sh builds each with the dialect the name gives
and checks it against its tracked manifest, and a failed build leaves out
exactly the recipes that need it (recipes.py select). No recipe names them
yet (task P17.49 adds the first). Nothing here runs a model or a device.
"""

from __future__ import annotations

import csv
import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "plans" / "runs" / "suite" / "run-suite.sh"
# Each new asset's need, as recipes.py prints it, and the dialect its build step passes to the tool.
NEEDS = {
    "context:ttir-ref-5f396fd6": "ttir", "context:d2m-ref-5f396fd6": "d2m",
    "retrieval:tt-mlir-ttir-tests": "ttir", "retrieval:tt-mlir-d2m-tests": "d2m",
}


def load(relative: str, name: str) -> ModuleType:
    """Import the module at REPO/`relative` under `name`."""
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def helper() -> ModuleType:
    """Return plans/runs/suite/recipes.py, the suite job's recipe helper."""
    return load("plans/runs/suite/recipes.py", "suite_recipes_mlir")


def case_body(label: str) -> str:
    """Return the text of run-suite.sh's build_asset case `label`, from its label line through its `;;`."""
    script = SCRIPT.read_text(encoding="utf-8")
    start = script.index(f"    {label})\n")
    return script[start:script.index(";;", start) + 2]


def test_a_recipe_naming_the_assets_needs_each_of_them() -> None:
    needs_of = helper().needs_of
    for dialect in ("ttir", "d2m"):
        data = {"context": [f"{dialect}-ref-5f396fd6"], "retrieval": {"index": f"tt-mlir-{dialect}-tests"}}
        assert needs_of(data) == [f"context:{dialect}-ref-5f396fd6", f"retrieval:tt-mlir-{dialect}-tests"]
        assert needs_of({"context": f"{dialect}-ref-5f396fd6"}) == [f"context:{dialect}-ref-5f396fd6"]


def test_each_need_names_a_dialect_the_tools_build() -> None:
    reference = load("tools/build_ttkernel_reference.py", "build_ttkernel_reference")
    rag = load("tools/build_ttkernel_rag.py", "build_ttkernel_rag")
    for need, dialect in NEEDS.items():
        kind, name = need.split(":", 1)
        assert name == (reference.DIALECTS[dialect].ref if kind == "context" else rag.CORPORA[dialect])
        tracked = REPO / "assets" / ("context" if kind == "context" else "rag") / name / "MANIFEST.yaml"
        assert tracked.is_file(), f"{need}: the build step checks the asset against its tracked manifest"


def test_run_suite_builds_each_asset_with_its_dialect_checks_it_and_logs_it_apart() -> None:
    references = case_body("context:ttir-ref-5f396fd6|context:d2m-ref-5f396fd6")
    indexes = case_body("retrieval:tt-mlir-ttir-tests|retrieval:tt-mlir-d2m-tests")
    assert 'build_ttkernel_reference.py build --tree "$mlir" --dialect "$dialect"' in references
    assert 'build_ttkernel_reference.py check --tree "$mlir" --dialect "$dialect"' in references
    assert "git diff --quiet HEAD -- assets/context" in references and '"$out/$dialect-ref.log"' in references
    assert "tools/fetch_bench.py assets/bench/tt-pairs-v0-ttkernel.yaml" in indexes, "the Rule 5 references first"
    assert 'build_ttkernel_rag.py build --dialect "$dialect" --tree "$mlir"' in indexes
    assert 'build_ttkernel_rag.py check --dialect "$dialect" --tree "$mlir"' in indexes
    assert "git diff --quiet HEAD -- assets/rag" in indexes and '"$out/$dialect-rag.log"' in indexes
    # The ttkernel steps of task P17.44 stay as they were: the tools' default dialect is ttkernel.
    assert "--dialect" not in case_body("context:ttkernel-ref-5f396fd6")
    assert "--dialect" not in case_body("retrieval:tt-mlir-tests")


@pytest.mark.skipif(shutil.which("bash") is None, reason="no bash on PATH")
@pytest.mark.parametrize(("need", "dialect"), sorted(NEEDS.items()))
def test_the_build_step_derives_the_dialect_from_the_need(need: str, dialect: str) -> None:
    label = ("context:ttir-ref-5f396fd6|context:d2m-ref-5f396fd6" if need.startswith("context:")
             else "retrieval:tt-mlir-ttir-tests|retrieval:tt-mlir-d2m-tests")
    lines = [line.strip().removeprefix("local mlir=\"$toolchains/tt-mlir@0.9.0.dev20260221/src/tt-mlir\" ")
             for line in case_body(label).splitlines() if re.match(r"\s*(local mlir=.* )?dialect=", line)]
    script = ("derive() {\n  local dialect\n" + "".join(f"  {line}\n" for line in lines) + '  echo "$dialect"\n}\n'
              f"derive '{need}'\n")
    # The script goes on stdin as bytes, so no newline is translated on any host.
    result = subprocess.run(["bash", "-s"], input=script.encode("ascii"), capture_output=True, check=True)
    assert result.stdout.decode("ascii").strip() == dialect


def test_a_failed_asset_leaves_out_exactly_the_recipes_that_need_it(tmp_path: Path,
                                                                    capsys: pytest.CaptureFixture[str]) -> None:
    module = helper()
    rows = {
        "ttir-ref": "context:ttir-ref-5f396fd6", "ttir-rag": "context:ttir-ref-5f396fd6,retrieval:tt-mlir-ttir-tests",
        "d2m-rag": "context:d2m-ref-5f396fd6,retrieval:tt-mlir-d2m-tests", "other": "-",
    }
    manifest = tmp_path / "recipes.tsv"
    with open(manifest, "w", encoding="ascii", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(module.HEADER)
        for name, needs in rows.items():
            writer.writerow(["sweepX", "model", name, "yes", "0" * 64, "SYNTHETIC/model", needs, f"{name}.yaml"])
    capsys.readouterr()
    assert module.main(["needs", "--manifest", str(manifest)]) == 0
    assert capsys.readouterr().out.split() == sorted(NEEDS)
    for failed, left_out in (("retrieval:tt-mlir-ttir-tests", ["ttir-rag"]),
                             ("context:ttir-ref-5f396fd6", ["ttir-rag", "ttir-ref"]),
                             ("context:d2m-ref-5f396fd6", ["d2m-rag"])):
        assert module.main(["select", "--manifest", str(manifest), "--sweep", "sweepX", "--model", "model",
                            "--failed", failed]) == 0
        count, pattern = capsys.readouterr().out.rstrip("\n").split("\t")
        assert int(count) == len(rows) - len(left_out)
        assert pattern == "^(" + "|".join(sorted(left_out)) + ")$"
