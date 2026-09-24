"""Tests for the ten-app lassi-hecbench-10 manifest, its support files, and item selection (task P1.2).

Bible: Benchmark Suites (lassi-hecbench-10, split rules), Source Papers
(LASSI quirk table: the entropy and PASS/FAIL rows), Harness Contract,
Design Principle 3, Agent Rule 5. The HeCBench pin and the source of every
file come from plans/spikes/p1-hecbench-pin.md (task P1.1).

The contract these tests encode:

- `assets/bench/lassi-hecbench-10.yaml` pins HeCBench at 692cba3 (the commit
  P1.1 chose) and lists the 10 apps of upstream LASSI's `*_main` files, each
  an eval item with `src/<app>-omp/main.cpp` and `src/<app>-cuda/main.cu`.
  Every model-facing file carries the sha256 of upstream's file
  (LanguageSources.sha256, file name -> hex digest). Each item records the
  run arguments of upstream's `experimental_setup` docstring exactly
  (SuiteItem.run_args; jacobi keeps one empty argument; the entry marked DNU
  is left out) and the languages that print PASS/FAIL (SuiteItem.passfail).
- Support files (SuiteItem.support, build-dir name -> path in the pinned
  sources; Suite.support_files reads them) are harness files: entropy's
  `reference.h` comes from `src/entropy-cuda/reference.h` of pinned HeCBench,
  and no other item has one. tools/fetch_bench.py fetches and checks them.
  Every build directory of the item holds them; tests/toolchains/
  test_harness_files.py checks that a model file cannot replace one.
- No upstream source includes a CUDA library outside the pinned archives
  (toolchains/cuda.pin COMPONENTS).
- A recipe selects items with `bench.items`; tests/fixtures/recipes/
  p0-smoke.yaml selects `layout` only, and an unknown item is refused.
- A `remote` test fetches the manifest on the build host and compiles all
  20 reference programs with the pinned toolchains in the compile sandbox,
  reporting 20/20 or each failure.

Oracles: the upstream checks read the pinned upstream files under
`third_party/LASSI` (fetched by tools/fetch_upstream.py) and first check each
file's git blob id against the ids the P1.1 spike logged (`git ls-tree` at
74b4681), so they compare with upstream's exact bytes. The notebook is parsed
with `ast` and `json`; no upstream code runs, and no upstream prompt or
notebook text is copied here (OQ-018). Without the upstream checkout those
tests skip with a reason that names the tool. The runner tests use the real
mock backend, stages, and none executor with fake toolchains, over small
synthetic sources (not HeCBench's) under a temporary bench root, read as
they are (tools/fetch_bench.py is what checks the sha256 of a fetch). No
value here is a measurement: blob ids and sha256 digests are content hashes.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

from lassi.bench import Direction, load_suite, sources_dir
from lassi.core.interfaces import BuildResult
from lassi.core.recipe import RecipeError, load_recipe
from lassi.core.registry import Registry
from lassi.core.runner import RunError, RunOptions, build_toolchain, run_recipe
from lassi.core.stages import CompileLoopStage, GenerateStage
from lassi.executors import NoneExecutor
from lassi.llm import MockBackend
from lassi.toolchains.pins import read_pin

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "assets" / "bench" / "lassi-hecbench-10.yaml"
SMOKE = REPO / "tests" / "fixtures" / "recipes" / "p0-smoke.yaml"
FETCH_BENCH = REPO / "tools" / "fetch_bench.py"

SUITE = "lassi-hecbench-10"
HECBENCH_REPO = "https://github.com/zjin-lcf/HeCBench"
# The HeCBench pin P1.1 chose: the newest commit on master's first-parent line holding all 20 upstream files
# byte for byte (plans/spikes/p1-hecbench-pin.md, Decision).
HECBENCH_PIN = "692cba32c5744f6ef024cca59f65e9488edba8bf"

UPSTREAM = REPO / "third_party" / "LASSI"
UPSTREAM_PIN = "74b46812523f2ff79b53b6880a4521690d7478b0"
UPSTREAM_MAINS = "translated_code/input_codes/HeCBench"
NOTEBOOK = "LASSI_pipeline_v0.ipynb"
FETCH_HINT = "run `uv run tools/fetch_upstream.py` to fetch upstream LASSI at 74b4681 into third_party/LASSI"

APPS = (
    "atomicCost",
    "bsearch",
    "colorwheel",
    "dense-embedding",
    "entropy",
    "jacobi",
    "layout",
    "matrix-rotate",
    "pathfinder",
    "randomAccess",
)
# HeCBench's file name per language, and the extension of upstream's copy.
HECBENCH_FILE = {"omp": "main.cpp", "cuda": "main.cu"}
UPSTREAM_EXT = {"omp": "cpp", "cuda": "cu"}
OTHER = {"omp": "cuda", "cuda": "omp"}

# Git blob ids of the 20 upstream files at 74b4681, from the P1.1 spike's `git ls-tree` output.
BLOBS = {
    ("atomicCost", "cuda"): "d0f36289ec57ae0b8d301373d65282c1f00d2663",
    ("atomicCost", "omp"): "7851d6da8d9f72074a5a78600b8abb09c9931f75",
    ("bsearch", "cuda"): "ba21f7cf1105dc3bfc4a4bccb606b6b27076c3b9",
    ("bsearch", "omp"): "bf485da7969bc5fd439a4ac12efb26501bc4e0a9",
    ("colorwheel", "cuda"): "714e36eb096f72bdb341bd581dbb3cf375cf8a93",
    ("colorwheel", "omp"): "cf865d1c607f428af9697688756fbbfa48d88091",
    ("dense-embedding", "cuda"): "3ee82f7ff5daf0cca3af269008edafefc1664be8",
    ("dense-embedding", "omp"): "0dd8a571a1fd612e9fc29c97368ee1dc099a361d",
    ("entropy", "cuda"): "3d54ed880fc8ce3ee6483bbb7414969fa6692234",
    ("entropy", "omp"): "273fb27df98db883e7180a3e4737b20981864974",
    ("jacobi", "cuda"): "778edd2abfb2386cb4d42696d672427ef1cb9987",
    ("jacobi", "omp"): "e920b53cb803e2f9f55e55520d7c269c50c993b8",
    ("layout", "cuda"): "2ee1a77eb1ce7542de152cd977b68f4bd7cf3ac5",
    ("layout", "omp"): "b0d64f2900734071965b8c9c84ab8d1f6dcfcd85",
    ("matrix-rotate", "cuda"): "8edcba8acc015d7e73b87c117413d2706602cac4",
    ("matrix-rotate", "omp"): "a67ddc5afebf1fe1bda90dbfec75299fc7b070f2",
    ("pathfinder", "cuda"): "18230cfe044ba3c35d2b2ad8bbd90ed8ae9c1bed",
    ("pathfinder", "omp"): "b298e20731fdd3a6d6bf365d48fe3611d841071e",
    ("randomAccess", "cuda"): "afc9d5088f3ad26d2fe8163759fcaf39b7eb676c",
    ("randomAccess", "omp"): "c038377f19db37086b6daf38726300b73fd11282",
}
# entropy's support file in pinned HeCBench and its blob id (P1.1 spike, Support files).
REFERENCE_H = "src/entropy-cuda/reference.h"
REFERENCE_H_BLOB = "112810d1560ceacac9e9d57a94285bfc54a2b1bf"

# The bible's quirk table, PASS/FAIL row: both languages for 7 of 10 apps, randomAccess CUDA only, none for
# bsearch and pathfinder.
BOTH = frozenset({"omp", "cuda"})
PASSFAIL = {
    "atomicCost": BOTH,
    "bsearch": frozenset(),
    "colorwheel": BOTH,
    "dense-embedding": BOTH,
    "entropy": BOTH,
    "jacobi": BOTH,
    "layout": BOTH,
    "matrix-rotate": BOTH,
    "pathfinder": frozenset(),
    "randomAccess": frozenset({"cuda"}),
}

HEX64 = re.compile(r"[0-9a-f]{64}")
QUOTED_INCLUDE = re.compile(r'^[ \t]*#[ \t]*include[ \t]*"([^"]+)"', re.MULTILINE)
ANGLE_INCLUDE = re.compile(r"^[ \t]*#[ \t]*include[ \t]*<([^>]+)>", re.MULTILINE)
# Header name prefixes of CUDA libraries whose archives toolchains/cuda.pin does not install (PHASE-NOTES P1,
# CUDA components): cuBLAS, cuRAND, cuFFT, cuSPARSE, cuSOLVER, NVRTC, nvJitLink, NPP, nvJPEG, cuDNN, NCCL, NVTX,
# and CUPTI.
OUTSIDE_PINNED_ARCHIVES = (
    "cublas", "curand", "cufft", "cusparse", "cusolver", "nvrtc", "nvjitlink",
    "npp", "nvjpeg", "cudnn", "nccl", "nvtx", "nvtoolsext", "cupti",
)  # fmt: skip
# One entry of upstream's experimental_setup docstring: `-- <app> [<args>]`, or `-- DNU -- <app> [<args>]`.
SETUP_ENTRY = re.compile(r"^\s*--\s+(?P<dnu>DNU\s+--\s+)?(?P<app>[A-Za-z][\w-]*)\s+(?P<args>\[[^\]]*\])")

# Synthetic stand-ins for the entropy sources and its support header (not HeCBench's text).
FAKE_OMP = '#include "reference.h"\nint main() {\n#pragma omp target\n  { }\n  return 0;\n}\n'
FAKE_CUDA = '#include "reference.h"\n__global__ void k() { }\nint main() {\n  k<<<1, 1>>>();\n  return 0;\n}\n'
FAKE_REFERENCE_H = "#pragma once\n// PLACEHOLDER synthetic support header for the P1.2 wiring test\n"
FAKE_LAYOUT_OMP = "int main() {\n#pragma omp target\n  { }\n  return 0;\n}\n"
FAKE_LAYOUT_CUDA = "__global__ void k() { }\nint main() {\n  k<<<1, 1>>>();\n  return 0;\n}\n"

REQUIRE = os.environ.get("LASSI_REQUIRE_SANDBOX") == "1"


# ---------------------------------------------------------------------------
# Helpers


def git_blob(data: bytes) -> str:
    """Return the git blob id of `data`, as `git hash-object` computes it."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def sha256(data: bytes) -> str:
    """Return the sha256 hex digest of `data`."""
    return hashlib.sha256(data).hexdigest()


def pinned_upstream() -> Path:
    """Return the upstream checkout; skip without it, and fail when a git checkout is not at the pin."""
    if not UPSTREAM.is_dir():
        pytest.skip(f"third_party/LASSI is missing; {FETCH_HINT}")
    if (UPSTREAM / ".git").exists():
        head = subprocess.run(
            ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip()
        assert head == UPSTREAM_PIN, f"third_party/LASSI is at {head or 'nothing'}, not {UPSTREAM_PIN}"
    return UPSTREAM


def upstream_main(app: str, language: str) -> bytes:
    """Return upstream's `*_main` file for `app` and `language`, checked against its logged blob id."""
    path = pinned_upstream() / UPSTREAM_MAINS / app / f"{app}-{language}_main.{UPSTREAM_EXT[language]}"
    if not path.is_file():
        pytest.skip(f"{path.relative_to(REPO).as_posix()} is missing; {FETCH_HINT}")
    data = path.read_bytes()
    assert git_blob(data) == BLOBS[(app, language)], f"{path} is not upstream's file at 74b4681"
    return data


def setup_entries() -> tuple[dict[str, list[str]], set[str]]:
    """Parse upstream's experimental_setup docstring: the kept entries (app -> args) and the apps marked DNU."""
    path = pinned_upstream() / NOTEBOOK
    if not path.is_file():
        pytest.skip(f"third_party/LASSI/{NOTEBOOK} is missing; {FETCH_HINT}")
    docstrings = []
    for cell in json.loads(path.read_bytes().decode("utf-8"))["cells"]:
        if cell.get("cell_type") != "code":
            continue
        try:
            tree = ast.parse("".join(cell["source"]))
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "experimental_setup":
                docstrings.append(ast.get_docstring(node, clean=False) or "")
    assert len(docstrings) == 1, f"expected one experimental_setup in the notebook, found {len(docstrings)}"
    kept: dict[str, list[str]] = {}
    dropped: set[str] = set()
    for line in docstrings[0].splitlines():
        match = SETUP_ENTRY.match(line)
        if match is None:
            continue
        if match["dnu"]:
            dropped.add(match["app"])
            continue
        args = ast.literal_eval(match["args"])
        assert isinstance(args, list) and all(isinstance(arg, str) for arg in args), line
        assert match["app"] not in kept, f"{match['app']} appears twice in experimental_setup"
        kept[match["app"]] = args
    return kept, dropped


def fetch_tool() -> ModuleType:
    """Import tools/fetch_bench.py by path and return the module."""
    spec = importlib.util.spec_from_file_location("fetch_bench", FETCH_BENCH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path: Path, text: str) -> Path:
    """Write `text` to `path` as UTF-8 with LF newlines, creating directories, and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))
    return path


# ---------------------------------------------------------------------------
# The manifest: pin, apps, and model-facing files


def test_manifest_pins_hecbench_at_the_commit_p1_1_chose() -> None:
    suite = load_suite(MANIFEST)
    assert suite.name == SUITE
    assert suite.repo == HECBENCH_REPO
    assert suite.commit == HECBENCH_PIN


def test_manifest_lists_the_ten_upstream_apps_as_eval_pairs_at_their_hecbench_paths() -> None:
    suite = load_suite(MANIFEST)
    assert sorted(suite.items) == sorted(APPS)
    for app, item in suite.items.items():
        assert item.split == "eval", f"{app}: lassi-hecbench-10 is eval only (bible Benchmark Suites; Agent Rule 5)"
        assert set(item.languages) == {"omp", "cuda"}, app
        for language, spec in item.languages.items():
            assert spec.dir == f"src/{app}-{language}", (app, language)
            assert tuple(spec.files) == (HECBENCH_FILE[language],), (app, language)


def test_every_model_facing_file_carries_a_sha256() -> None:
    suite = load_suite(MANIFEST)
    for app, item in suite.items.items():
        for language, spec in item.languages.items():
            digests = dict(spec.sha256)
            assert set(digests) == set(spec.files), (app, language, digests)
            assert all(HEX64.fullmatch(value) for value in digests.values()), (app, language, digests)


@pytest.mark.parametrize("app", APPS)
def test_each_sha256_is_that_of_upstreams_file(app: str) -> None:
    item = load_suite(MANIFEST).items[app]
    for language, spec in item.languages.items():
        data = upstream_main(app, language)
        assert dict(spec.sha256) == {HECBENCH_FILE[language]: sha256(data)}, (app, language)


# ---------------------------------------------------------------------------
# Run arguments and PASS/FAIL


def test_every_item_records_its_run_arguments_and_jacobi_keeps_one_empty_argument() -> None:
    suite = load_suite(MANIFEST)
    for app, item in suite.items.items():
        args = tuple(item.run_args)
        assert args and all(isinstance(arg, str) for arg in args), (app, item.run_args)
    assert tuple(suite.items["jacobi"].run_args) == ("",)


def test_run_arguments_are_upstreams_experimental_setup_entries_without_the_dnu_one() -> None:
    kept, dropped = setup_entries()
    suite = load_suite(MANIFEST)
    assert dropped, "the docstring marks one entry DNU"
    assert not dropped & set(suite.items), f"entries marked DNU are left out: {sorted(dropped & set(suite.items))}"
    assert set(kept) == set(suite.items) == set(APPS)
    for app, args in kept.items():
        assert list(suite.items[app].run_args) == args, app


def test_passfail_languages_follow_the_bible_quirk_table() -> None:
    suite = load_suite(MANIFEST)
    assert {app: frozenset(item.passfail) for app, item in suite.items.items()} == PASSFAIL


@pytest.mark.parametrize("app", APPS)
def test_passfail_languages_are_those_whose_upstream_source_prints_pass(app: str) -> None:
    item = load_suite(MANIFEST).items[app]
    printing = {language for language in HECBENCH_FILE if b'"PASS' in upstream_main(app, language)}
    assert frozenset(item.passfail) == printing


# ---------------------------------------------------------------------------
# Support files and CUDA components


def test_entropy_takes_reference_h_from_pinned_hecbench_and_no_other_item_has_support_files() -> None:
    suite = load_suite(MANIFEST)
    assert dict(suite.items["entropy"].support) == {"reference.h": REFERENCE_H}
    others = {app: dict(item.support) for app, item in suite.items.items() if app != "entropy"}
    assert all(not support for support in others.values()), others


@pytest.mark.parametrize("app", APPS)
def test_support_files_are_exactly_the_local_headers_upstream_includes(app: str) -> None:
    item = load_suite(MANIFEST).items[app]
    included = set()
    for language in HECBENCH_FILE:
        included |= set(QUOTED_INCLUDE.findall(upstream_main(app, language).decode("utf-8")))
    assert set(item.support) == included


def test_fetch_lists_the_support_file_with_the_item_files(tmp_path: Path) -> None:
    fetch_bench = fetch_tool()
    suite = load_suite(MANIFEST)
    item_dirs = {f"src/{app}-{language}" for app in APPS for language in HECBENCH_FILE}
    sparse = fetch_bench.sparse_dirs(suite)
    assert item_dirs <= set(sparse)
    assert all(any(entry == d or entry.startswith(f"{d}/") for d in item_dirs) for entry in sparse), sparse
    missing = fetch_bench.missing_files(suite, tmp_path / "empty")
    mains = [f"src/{app}-{language}/{HECBENCH_FILE[language]}" for app in APPS for language in HECBENCH_FILE]
    assert sorted(missing) == sorted([*mains, REFERENCE_H])


def test_no_upstream_source_includes_a_cuda_library_outside_the_pinned_archives() -> None:
    components = [name.lower() for name in read_pin("cuda")["COMPONENTS"].split()]
    assert components, "toolchains/cuda.pin names its archives in COMPONENTS"
    pinned = [prefix for prefix in OUTSIDE_PINNED_ARCHIVES for name in components if prefix in name]
    assert not pinned, f"the denylist names a pinned archive: {pinned}"
    for app in APPS:
        for language in HECBENCH_FILE:
            headers = ANGLE_INCLUDE.findall(upstream_main(app, language).decode("utf-8"))
            outside = [name for name in headers if name.lower().startswith(OUTSIDE_PINNED_ARCHIVES)]
            assert not outside, (app, language, outside)


# ---------------------------------------------------------------------------
# Item selection


def test_p0_smoke_selects_layout_only() -> None:
    recipe = load_recipe(SMOKE)
    assert recipe.data["bench"]["suite"] == SUITE
    assert recipe.data["bench"]["items"] == ["layout"]


class _Recorded:
    """The build directories the fake toolchains were given, in order."""

    workdirs: list[Path] = []


def harness_fake(registered_as: str) -> type:
    """Return a Toolchain class without PIN that writes the model's files, then any harness files, and builds.

    It accepts harness files as a `harness` mapping (relative path -> text)
    and writes them after the model's files, so a harness file always wins;
    files the stage passes inside `files` or writes into the build directory
    itself land as they are. It writes a placeholder artifact and runs no
    command.
    """

    class HarnessFake:
        """A fake Toolchain, built as factory(); it writes files and compiles nothing."""

        name = registered_as
        capabilities = frozenset({"diagnostics"})

        def build(
            self, files: Mapping[str, str], workdir: Path, harness: Mapping[str, str] | None = None
        ) -> BuildResult:
            """Write `files`, then `harness`, under `workdir`; return a placeholder artifact."""
            workdir = Path(workdir)
            _Recorded.workdirs.append(workdir)
            for path, text in [*files.items(), *(harness or {}).items()]:
                write(workdir / path, text)
            artifact = workdir / "main"
            artifact.write_bytes(b"PLACEHOLDER artifact of the fake toolchain\n")
            return BuildResult(artifact=artifact, diagnostics=[])

    return HarnessFake


def fake_registry() -> Registry:
    """Return a Registry with the real mock backend, stages, and none executor, and harness-aware fake toolchains."""
    registry = Registry()
    registry.register("LLMBackend", "mock", MockBackend)
    registry.register("Stage", "generate", GenerateStage)
    registry.register("Stage", "compile_loop", CompileLoopStage)
    registry.register("Executor", "none", NoneExecutor)
    registry.register("Toolchain", "nvcc-sm80", harness_fake("nvcc-sm80"))
    registry.register("Toolchain", "nvcpp-cc80", harness_fake("nvcpp-cc80"))
    return registry


def recipe_data(items: list[str], directions: list[dict[str, str]]) -> dict[str, Any]:
    """Return a compile-only mock recipe over lassi-hecbench-10 that selects `items`."""
    return {
        "extends": "base",
        "model": {"backend": "mock", "id": "mock-reference"},
        "llm": {"sampling": {"max_tokens": 4096}},
        "bench": {"suite": SUITE, "split": "eval", "items": items},
        "directions": directions,
        "prompts": "p0-smoke",
        "toolchain": {"cuda": "nvcc-sm80", "omp": "nvcpp-cc80"},
        "stages": ["generate", "compile_loop"],
        "executor": {"kind": "none"},
        "trials": {"n": 1},
    }


@pytest.fixture
def clean_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Unset the gate variables, point TMPDIR at a test directory, reset the build log; return the runs root."""
    for name in ("LASSI_RUNS_ROOT", "LASSI_SCRATCH", "LASSI_TOOLCHAINS", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"):
        monkeypatch.delenv(name, raising=False)
    tmpdir = tmp_path / "compile-tmp"
    tmpdir.mkdir()
    monkeypatch.setenv("TMPDIR", str(tmpdir))
    _Recorded.workdirs = []
    return tmp_path / "runs-root"


def run_with(tmp_path: Path, runs_root: Path, name: str, data: dict[str, Any], bench_root: Path) -> Path:
    """Write `data` as the recipe `<name>.yaml` under tmp_path, run it with the fake registry, return the run dir."""
    recipe = tmp_path / f"{name}.yaml"
    recipe.write_bytes(yaml.safe_dump(data, sort_keys=False).encode("ascii"))
    options = RunOptions(runs_root=runs_root, run_id="x", bench_root=bench_root, registry=fake_registry())
    return run_recipe(recipe, options)


def test_a_recipe_runs_only_the_items_it_selects(tmp_path: Path, clean_env: Path) -> None:
    # Only layout's sources exist under the bench root, so a run that reached any other item would fail.
    bench_root = tmp_path / "bench"
    write(bench_root / "src/layout-omp/main.cpp", FAKE_LAYOUT_OMP)
    write(bench_root / "src/layout-cuda/main.cu", FAKE_LAYOUT_CUDA)
    data = recipe_data(["layout"], [{"source": "omp", "target": "cuda"}])
    run_dir = run_with(tmp_path, clean_env, "select-layout", data, bench_root)
    trials = sorted(path.relative_to(run_dir).as_posix() for path in run_dir.rglob("trial.json"))
    assert trials == [f"select-layout/mock-reference/{SUITE}/omp-cuda/layout/run01/trial.json"]


def test_a_recipe_selecting_an_unknown_item_is_refused(tmp_path: Path, clean_env: Path) -> None:
    bench_root = tmp_path / "bench"
    write(bench_root / "src/layout-omp/main.cpp", FAKE_LAYOUT_OMP)
    write(bench_root / "src/layout-cuda/main.cu", FAKE_LAYOUT_CUDA)
    data = recipe_data(["layout", "no-such-app"], [{"source": "omp", "target": "cuda"}])
    with pytest.raises((RunError, RecipeError), match="no-such-app"):
        run_with(tmp_path, clean_env, "select-unknown", data, bench_root)
    assert not clean_env.exists() or not list(clean_env.rglob("trial.json"))


def test_support_files_land_in_every_build_directory_of_their_item(tmp_path: Path, clean_env: Path) -> None:
    bench_root = tmp_path / "bench"
    write(bench_root / "src/entropy-omp/main.cpp", FAKE_OMP)
    write(bench_root / "src/entropy-cuda/main.cu", FAKE_CUDA)
    write(bench_root / REFERENCE_H, FAKE_REFERENCE_H)
    directions = [{"source": "omp", "target": "cuda"}, {"source": "cuda", "target": "omp"}]
    run_dir = run_with(tmp_path, clean_env, "entropy-support", recipe_data(["entropy"], directions), bench_root)
    builds = sorted(path for path in run_dir.rglob("build") if path.is_dir() and path.parent.name.startswith("attempt"))
    assert len(builds) == 2, builds
    assert sorted(path.resolve() for path in builds) == sorted({path.resolve() for path in _Recorded.workdirs})
    for build in builds:
        assert (build / "reference.h").read_bytes() == FAKE_REFERENCE_H.encode("utf-8"), build


# ---------------------------------------------------------------------------
# Remote: fetch the manifest and compile all 20 reference programs


def host_problem() -> str:
    """Return why this host cannot fetch and compile the suite, or "" when it can."""
    if not sys.platform.startswith("linux"):
        return f"needs the build host (Linux), not {sys.platform}; run it through `uv run tools/rx.py run`"
    for name in ("LASSI_SCRATCH", "LASSI_RUNS_ROOT", "LASSI_TOOLCHAINS", "TMPDIR"):
        if not os.environ.get(name):
            return f"{name} is not set; run it through `uv run tools/rx.py run`"
    return ""


def compile_report(suite: Any, root: Path, base: Path) -> tuple[list[str], list[str]]:
    """Compile each item's reference in both languages with the pinned presets; return report lines and failures."""
    lines: list[str] = []
    failures: list[str] = []
    toolchains = Path(os.environ["LASSI_TOOLCHAINS"])
    for preset, target in (("nvcc-sm80", "cuda"), ("nvcpp-cc80", "omp")):
        built = build_toolchain(preset, toolchains)
        direction = Direction(source=OTHER[target], target=target)
        for app in APPS:
            workdir = base / f"{app}-{target}" / "build"
            workdir.mkdir(parents=True)
            files = suite.reference_target(app, direction, root, purpose="eval")
            harness = suite.support_files(app, root, purpose="eval")
            result = built.toolchain.build(files, workdir, harness=harness)
            if result.artifact is not None and result.artifact.is_file():
                lines.append(f"ok   {app} {target} ({preset})")
                continue
            errors = [f"{d.file}:{d.line} {d.message}"[:200] for d in result.diagnostics if d.severity == "error"]
            line = f"FAIL {app} {target} ({preset}): {'; '.join(errors[:3]) or 'no error parsed'}"
            lines.append(line)
            failures.append(line)
    return lines, failures


@pytest.mark.remote
@pytest.mark.slow
def test_all_20_reference_programs_compile_with_the_pinned_toolchains() -> None:
    # Compiling a reference is no training or tuning on the eval items (Agent Rule 5); nothing runs.
    problem = host_problem()
    if problem:
        if REQUIRE:
            pytest.fail(f"LASSI_REQUIRE_SANDBOX=1, but {problem}")
        pytest.skip(problem)
    assert fetch_tool().main([str(MANIFEST)]) == 0, "tools/fetch_bench.py failed"
    suite = load_suite(MANIFEST)
    root = sources_dir(Path(os.environ["LASSI_SCRATCH"]), suite)
    for app in APPS:
        for language, spec in suite.items[app].languages.items():
            data = (root / spec.dir / HECBENCH_FILE[language]).read_bytes()
            assert git_blob(data) == BLOBS[(app, language)], (app, language)
            assert sha256(data) == dict(spec.sha256)[HECBENCH_FILE[language]], (app, language)
    assert git_blob((root / REFERENCE_H).read_bytes()) == REFERENCE_H_BLOB
    base = Path(tempfile.mkdtemp(prefix="lassi-hecbench10-compile.", dir=os.environ["LASSI_RUNS_ROOT"]))
    try:
        lines, failures = compile_report(suite, root, base)
    finally:
        shutil.rmtree(base)
    compiled = 2 * len(APPS) - len(failures)
    lines.append(f"compiled {compiled}/{2 * len(APPS)} reference programs of {SUITE} at HeCBench {suite.commit}")
    report = "\n".join(lines)
    print(report)
    assert not failures, report
