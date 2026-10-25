"""Tests for the lassi-ee-22 manifest: its 22 apps, absent languages, and fetch (task P9.7).

Bible: Benchmark Suites (lassi-ee-22, split rules), Source Papers (LASSI-EE:
the 22 apps), Oracles (stdout_mask, passfail), Harness Contract, Agent Rule 5.
OQ-089 (a): the CPU port runs the 18 apps that have an OpenMP version in
HeCBench at 692cba3; marchingCubes, segment-reduce, and jaccard (no OpenMP
version) and miniMDock (not in HeCBench) are listed and named as not ported.

The contract these tests encode:

- `assets/bench/lassi-ee-22.yaml` pins HeCBench at 692cba3, the commit of
  lassi-hecbench-10, and lists the 22 apps of the paper's Table I
  (plans/spikes/p5-v0-corpus.md, section 4), each split eval. XSBench is
  HeCBench's xsbench, so its directories are src/xsbench-<language>.
- Languages: each of omp, cuda, and hip that HeCBench holds at the pin as
  src/<directory>-<language> (HeCBench's GitHub tree at 692cba3, read
  2026-10-25). Eighteen items have all three. marchingCubes,
  segment-reduce, and jaccard have cuda and hip and declare omp absent;
  miniMDock has none and declares all three absent (SuiteItem.absent, a
  note per language). Reading an absent language raises ValueError quoting
  its note.
- Files and support: a language's `files` are its directory's own sources
  (every .cpp, .cu, and .h file there but reference.h), each with a sha256.
  Its `support` holds each file its build takes from another directory
  (HeCBench's -I../<app>-cuda and ../<app>-cuda/ sources) and reference.h,
  the CPU reference check, in every language, each {upstream, sha256}. No
  item declares item-level support.
- run_args: the arguments of each app's Makefile `run` target at the pin,
  the same in every language ([DESIGN]).
- passfail: the languages whose sources print PASS as a whole word, the
  token the passfail oracle reads.
- tools/fetch_bench.py lists every file and support file and checks a
  sha256 for each, so a changed file is refused, with no change of its own.

Sources-backed checks read the pinned sources under
$LASSI_SCRATCH/bench/lassi-ee-22@<commit> (tools/fetch_bench.py puts them
there) and skip without them; they only read the sources, never build or run
them. No HeCBench text is copied here. No value here is a measurement:
digests are content hashes.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import re
from pathlib import Path
from types import ModuleType

import pytest

from lassi.bench import Direction, Suite, load_suite, sources_dir

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "assets" / "bench" / "lassi-ee-22.yaml"
HECBENCH10 = REPO / "assets" / "bench" / "lassi-hecbench-10.yaml"
FETCH_BENCH = REPO / "tools" / "fetch_bench.py"

SUITE = "lassi-ee-22"
HECBENCH_REPO = "https://github.com/zjin-lcf/HeCBench"
HECBENCH_PIN = "692cba32c5744f6ef024cca59f65e9488edba8bf"

ALL = frozenset({"omp", "cuda", "hip"})
GPU = frozenset({"cuda", "hip"})
NONE: frozenset[str] = frozenset()
# The paper's 22 apps (Table I) and the languages HeCBench holds for each at the pin.
LANGUAGES = {
    "all-pairs-distance": ALL,
    "bsearch": ALL,
    "chacha20": ALL,
    "colorwheel": ALL,
    "dense-embedding": ALL,
    "entropy": ALL,
    "extrema": ALL,
    "floydwarshall": ALL,
    "jaccard": GPU,
    "jacobi": ALL,
    "keogh": ALL,
    "layout": ALL,
    "lid-driven-cavity": ALL,
    "marchingCubes": GPU,
    "matrix-rotate": ALL,
    "miniMDock": NONE,
    "murmurhash3": ALL,
    "pathfinder": ALL,
    "randomAccess": ALL,
    "segment-reduce": GPU,
    "threadfence": ALL,
    "XSBench": ALL,
}
PORTED = sorted(app for app, languages in LANGUAGES.items() if "omp" in languages)
NOT_PORTED = {"marchingCubes": {"omp"}, "segment-reduce": {"omp"}, "jaccard": {"omp"}, "miniMDock": set(ALL)}
# HeCBench's directory name where it differs from the app's name.
DIRECTORY = {"XSBench": "xsbench"}

# The arguments of each app's Makefile `run` target at the pin (the same in every language).
RUN_ARGS = {
    "all-pairs-distance": ["10000"],
    "bsearch": ["8388608", "100"],
    "chacha20": ["100000"],
    "colorwheel": ["10", "4096", "100"],
    "dense-embedding": ["100000", "256", "1000"],
    "entropy": ["8192", "8192", "100"],
    "extrema": ["100"],
    "floydwarshall": ["1024", "100", "16"],
    "jaccard": ["1024", "512", "1000"],
    "jacobi": [],
    "keogh": ["256", "20000000", "100"],
    "layout": ["1000"],
    "lid-driven-cavity": [],
    "marchingCubes": ["100"],
    "matrix-rotate": ["5000", "100"],
    "miniMDock": [],
    "murmurhash3": ["100000", "100"],
    "pathfinder": ["100000", "1000", "5"],
    "randomAccess": ["10"],
    "segment-reduce": ["16384", "100"],
    "threadfence": ["100", "10000000"],
    "XSBench": ["-s", "large", "-m", "event", "-r", "10"],
}

# The languages whose sources print PASS as a whole word. murmurhash3 prints SUCCESS, extrema only FAILED, and
# jaccard's HIP version PASSED and FAILED, none of them the PASS token; randomAccess's OpenMP version prints none.
PASSFAIL = {
    "all-pairs-distance": ALL,
    "chacha20": ALL,
    "colorwheel": ALL,
    "dense-embedding": ALL,
    "entropy": ALL,
    "floydwarshall": ALL,
    "jaccard": frozenset({"cuda"}),
    "jacobi": ALL,
    "keogh": ALL,
    "layout": ALL,
    "marchingCubes": GPU,
    "matrix-rotate": ALL,
    "randomAccess": GPU,
    "threadfence": ALL,
}

# Each language's own files: main.cpp (omp) or main.cu (cuda, hip) unless listed here.
XSBENCH_CUDA = ("GridInit.cu", "Main.cu", "Materials.cu", "Simulation.cu", "XSutils.cu", "XSbench_header.h", "io.cu")
LID_HEADERS = (
    "black_kernel.h", "calc_residual.h", "calculate_F.h", "calculate_G.h", "calculate_u.h", "calculate_v.h",
    "red_kernel.h", "set_BCs.h", "set_horz_pres_BCs.h", "set_vert_pres_BCs.h", "sum_pressure.h",
)  # fmt: skip
FILES = {
    ("chacha20", "omp"): ("main.cpp", "chacha20.h"),
    ("chacha20", "cuda"): ("main.cu", "chacha20.h"),
    ("lid-driven-cavity", "omp"): ("main.cpp", *LID_HEADERS),
    ("marchingCubes", "cuda"): ("main.cu", "tables.h"),
    ("murmurhash3", "omp"): ("murmurhash3.cpp",),
    ("murmurhash3", "cuda"): ("murmurhash3.cu",),
    ("murmurhash3", "hip"): ("murmurhash3.cu",),
    ("XSBench", "omp"): ("Simulation.cpp",),
    ("XSBench", "cuda"): XSBENCH_CUDA,
    ("XSBench", "hip"): ("Simulation.cu",),
}
MAIN = {"omp": ("main.cpp",), "cuda": ("main.cu",), "hip": ("main.cu",)}

# Support files (build-directory name -> path in the pinned sources), per item and language; none elsewhere.
XSBENCH_SHARED = {
    name: f"src/xsbench-cuda/{name}"
    for name in ("GridInit.cu", "Main.cu", "Materials.cu", "XSutils.cu", "XSbench_header.h", "io.cu")
}
SUPPORT = {
    ("chacha20", "hip"): {"chacha20.h": "src/chacha20-cuda/chacha20.h"},
    **{("entropy", language): {"reference.h": "src/entropy-cuda/reference.h"} for language in ALL},
    **{("keogh", language): {"reference.h": "src/keogh-cuda/reference.h"} for language in ALL},
    ("marchingCubes", "hip"): {"tables.h": "src/marchingCubes-cuda/tables.h"},
    ("XSBench", "omp"): XSBENCH_SHARED,
    ("XSBench", "hip"): XSBENCH_SHARED,
}

HEX64 = re.compile(r"[0-9a-f]{64}")
SOURCE_SUFFIXES = (".cpp", ".cu", ".h")
QUOTED_INCLUDE = re.compile(r'^[ \t]*#[ \t]*include[ \t]*"([^"]+)"', re.MULTILINE)
# A string literal holding PASS as a whole word, the token the passfail oracle reads (lassi.oracles PASS_TOKEN).
PASS_LITERAL = re.compile(r'"[^"\n]*\bPASS\b[^"\n]*"')
# A Makefile run target: `run: $(program)`, then one recipe line `$(LAUNCHER) ./$(program) <arguments>`.
RUN_TARGET = re.compile(r"^run:[^\n]*\n\t[ \t]*\$\(LAUNCHER\)[ \t]+\./\$\(program\)([^\n]*)$", re.MULTILINE)


# ---------------------------------------------------------------------------
# Helpers


def suite() -> Suite:
    """Return the loaded lassi-ee-22 manifest."""
    return load_suite(MANIFEST)


def directory(app: str, language: str) -> str:
    """Return the app's HeCBench directory for `language`."""
    return f"src/{DIRECTORY.get(app, app)}-{language}"


def expected_files(app: str, language: str) -> tuple[str, ...]:
    """Return the files the manifest lists for one language of an app."""
    return FILES.get((app, language), MAIN[language])


def fetch_tool() -> ModuleType:
    """Import tools/fetch_bench.py by path and return the module."""
    spec = importlib.util.spec_from_file_location("fetch_bench", FETCH_BENCH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path: Path, data: bytes) -> None:
    """Write `data` to `path`, creating directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


@pytest.fixture(scope="module")
def pinned_root() -> Path:
    """Return the fetched pinned sources; skip without them, and fail when a file is missing or differs."""
    scratch = os.environ.get("LASSI_SCRATCH", "")
    if not scratch:
        pytest.skip("LASSI_SCRATCH is not set; run `uv run tools/fetch_bench.py assets/bench/lassi-ee-22.yaml` there")
    root = sources_dir(Path(scratch), suite())
    if not root.is_dir():
        pytest.skip(f"{root} is missing; run `uv run tools/fetch_bench.py assets/bench/lassi-ee-22.yaml`")
    fetch_bench = fetch_tool()
    assert fetch_bench.missing_files(suite(), root) == [], "a listed file is missing from the fetched sources"
    assert fetch_bench.mismatched_files(suite(), root) == [], "a fetched file differs from its sha256"
    return root


def build_files(app: str, language: str, root: Path) -> dict[str, str]:
    """Return what one language's build directory holds from the pinned sources: its files, then its support."""
    spec = suite().items[app].languages[language]
    texts = {name: (root / spec.dir / name).read_bytes().decode("utf-8") for name in spec.files}
    texts.update({name: (root / entry.path).read_bytes().decode("utf-8") for name, entry in spec.support.items()})
    return texts


# ---------------------------------------------------------------------------
# The manifest: pin, apps, splits, languages


def test_manifest_pins_hecbench_at_the_lassi_hecbench_10_commit() -> None:
    loaded = suite()
    assert loaded.name == SUITE
    assert loaded.repo == HECBENCH_REPO
    assert loaded.commit == HECBENCH_PIN == load_suite(HECBENCH10).commit
    assert loaded.pin is None


def test_manifest_lists_the_22_apps_of_the_paper_each_in_the_eval_split() -> None:
    loaded = suite()
    assert sorted(loaded.items) == sorted(LANGUAGES)
    assert len(loaded.items) == 22
    for app, item in loaded.items.items():
        assert item.split == "eval", f"{app}: lassi-ee-22 is eval only (bible Benchmark Suites; Agent Rule 5)"


def test_each_item_has_the_languages_hecbench_holds_at_the_pin_in_its_directories() -> None:
    for app, item in suite().items.items():
        assert set(item.languages) == LANGUAGES[app], app
        for language, spec in item.languages.items():
            assert spec.dir == directory(app, language), (app, language)
            assert not spec.tracked, (app, language)


def test_the_18_ported_items_have_an_omp_version_with_files_and_a_sha256_for_each() -> None:
    assert len(PORTED) == 18
    loaded = suite()
    for app in PORTED:
        for language, spec in loaded.items[app].languages.items():
            assert spec.files == expected_files(app, language), (app, language)
            assert set(spec.sha256) == set(spec.files), (app, language)
            assert all(HEX64.fullmatch(digest) for digest in spec.sha256.values()), (app, language)


def test_the_unported_items_list_their_gpu_files_with_a_sha256_for_each() -> None:
    loaded = suite()
    for app in ("marchingCubes", "segment-reduce", "jaccard"):
        for language, spec in loaded.items[app].languages.items():
            assert spec.files == expected_files(app, language), (app, language)
            assert set(spec.sha256) == set(spec.files), (app, language)


def test_the_four_unported_items_declare_each_absent_language_with_a_note() -> None:
    loaded = suite()
    absent = {app: set(item.absent) for app, item in loaded.items.items() if item.absent}
    assert absent == NOT_PORTED
    for app in NOT_PORTED:
        for language, note in loaded.items[app].absent.items():
            assert isinstance(note, str) and note.strip(), (app, language)
            assert "OQ-089" in note, f"{app}/{language}: the note names the owner's decision"
    assert set(loaded.items["miniMDock"].languages) == set()


@pytest.mark.parametrize("app", sorted(NOT_PORTED))
def test_reading_an_absent_omp_version_raises_quoting_its_note(app: str, tmp_path: Path) -> None:
    loaded = suite()
    note = loaded.items[app].absent["omp"]
    with pytest.raises(ValueError, match=re.escape(note)):
        loaded.source_files(app, Direction("omp", "omp"), tmp_path, purpose="eval")
    with pytest.raises(ValueError, match=re.escape(note)):
        loaded.support_files(app, tmp_path, purpose="eval", language="omp")


def test_every_item_stays_refused_to_training_prompt_tuning_and_harvest() -> None:
    from lassi.bench import EvalSplitError

    loaded = suite()
    for app in loaded.items:
        for purpose in ("train", "prompt-tuning", "harvest"):
            with pytest.raises(EvalSplitError, match=app):
                loaded.item(app, purpose=purpose)
    assert loaded.train_view().train_items == {}


# ---------------------------------------------------------------------------
# Run arguments, PASS/FAIL, and support files


def test_run_args_are_lists_of_strings_from_each_makefile_run_target() -> None:
    for app, item in suite().items.items():
        assert isinstance(item.run_args, tuple), app
        assert all(isinstance(arg, str) and arg for arg in item.run_args), (app, item.run_args)
        assert list(item.run_args) == RUN_ARGS[app], app


def test_passfail_lists_the_languages_that_print_the_pass_token() -> None:
    for app, item in suite().items.items():
        assert frozenset(item.passfail) == PASSFAIL.get(app, NONE), app


def test_support_files_are_the_cross_directory_files_and_reference_h_each_with_a_sha256() -> None:
    for app, item in suite().items.items():
        assert dict(item.support) == {}, f"{app}: support is per language here, so each file has a sha256"
        for language, spec in item.languages.items():
            found = {name: entry.path for name, entry in spec.support.items()}
            assert found == SUPPORT.get((app, language), {}), (app, language)
            for name, entry in spec.support.items():
                assert not entry.tracked and HEX64.fullmatch(entry.sha256 or ""), (app, language, name)


def test_a_support_file_named_in_two_languages_has_one_digest() -> None:
    digests: dict[str, set[str]] = {}
    for item in suite().items.values():
        for spec in item.languages.values():
            for entry in spec.support.values():
                digests.setdefault(entry.path, set()).add(entry.sha256 or "")
            for name, digest in spec.sha256.items():
                digests.setdefault(f"{spec.dir}/{name}", set()).add(digest)
    assert {path: found for path, found in digests.items() if len(found) != 1} == {}


def test_items_shared_with_lassi_hecbench_10_carry_its_digests() -> None:
    ours, theirs = suite(), load_suite(HECBENCH10)
    shared = sorted(set(ours.items) & set(theirs.items))
    assert shared == [
        "bsearch", "colorwheel", "dense-embedding", "entropy", "jacobi", "layout", "matrix-rotate", "pathfinder",
        "randomAccess",
    ]  # fmt: skip
    for app in shared:
        for language in ("omp", "cuda"):
            mine, other = ours.items[app].languages[language], theirs.items[app].languages[language]
            assert (mine.dir, dict(mine.sha256)) == (other.dir, dict(other.sha256)), (app, language)


# ---------------------------------------------------------------------------
# tools/fetch_bench.py: every file fetched and checked


def test_fetch_checks_out_the_item_directories_and_lists_every_file() -> None:
    fetch_bench = fetch_tool()
    loaded = suite()
    dirs = {directory(app, language) for app, languages in LANGUAGES.items() for language in languages}
    assert fetch_bench.sparse_dirs(loaded) == sorted(dirs), "every support file lies in an item directory"
    files = [f"{directory(app, lang)}/{name}" for app, ls in LANGUAGES.items() for lang in ls
             for name in expected_files(app, lang)]  # fmt: skip
    support = {path for found in SUPPORT.values() for path in found.values()}
    assert sorted(fetch_bench.listed_files(loaded)) == sorted(set(files) | support)


def test_fetch_refuses_every_listed_file_that_differs_from_its_sha256(tmp_path: Path) -> None:
    fetch_bench = fetch_tool()
    loaded = suite()
    listed = fetch_bench.listed_files(loaded)
    assert sorted(fetch_bench.missing_files(loaded, tmp_path / "empty")) == sorted(listed)
    changed = tmp_path / "changed"
    for path in listed:
        write(changed / path, b"// SYNTHETIC: not the pinned file\n")
    assert sorted(fetch_bench.mismatched_files(loaded, changed)) == sorted(listed), "every file has a sha256"


def test_fetch_keeps_a_file_whose_bytes_match_and_names_only_the_one_that_changed(tmp_path: Path) -> None:
    # A synthetic manifest of the same shape, whose digests are those of synthetic bytes written here.
    good, bad = b"// SYNTHETIC main\n", b"// SYNTHETIC header\n"
    text = (
        "suite: tiny-ee\n"
        f"repo: {HECBENCH_REPO}\n"
        f"commit: {HECBENCH_PIN}\n"
        "items:\n"
        "  one:\n"
        "    split: eval\n"
        "    absent: {cuda: 'SYNTHETIC note'}\n"
        "    languages:\n"
        "      omp:\n"
        "        dir: src/one-omp\n"
        "        files: [main.cpp]\n"
        f"        sha256: {{main.cpp: {hashlib.sha256(good).hexdigest()}}}\n"
        f"        support: {{ref.h: {{upstream: src/one-gpu/ref.h, sha256: {hashlib.sha256(bad).hexdigest()}}}}}\n"
    )
    manifest = tmp_path / "tiny-ee.yaml"
    manifest.write_bytes(text.encode("ascii"))
    fetch_bench = fetch_tool()
    tiny = load_suite(manifest)
    root = tmp_path / "sources"
    write(root / "src/one-omp/main.cpp", good)
    write(root / "src/one-gpu/ref.h", b"// SYNTHETIC header, changed\n")
    assert fetch_bench.missing_files(tiny, root) == []
    assert fetch_bench.mismatched_files(tiny, root) == ["src/one-gpu/ref.h"]


# ---------------------------------------------------------------------------
# Against the pinned sources (skipped unless fetched)


def test_the_fetched_sources_match_every_sha256(pinned_root: Path) -> None:
    fetch_bench = fetch_tool()
    assert fetch_bench.missing_files(suite(), pinned_root) == []
    assert fetch_bench.mismatched_files(suite(), pinned_root) == []


def test_files_are_each_directorys_own_sources_but_reference_h(pinned_root: Path) -> None:
    for app, item in suite().items.items():
        for language, spec in item.languages.items():
            here = pinned_root / spec.dir
            own = {path.name for path in here.iterdir() if path.suffix in SOURCE_SUFFIXES} - {"reference.h"}
            assert set(spec.files) == own, (app, language)


def test_every_quoted_include_is_a_file_or_support_file_of_its_language(pinned_root: Path) -> None:
    for app, item in suite().items.items():
        for language in item.languages:
            texts = build_files(app, language, pinned_root)
            included = {name for text in texts.values() for name in QUOTED_INCLUDE.findall(text)}
            included.discard("papi.h")  # behind #ifdef PAPI in XSbench_header.h; no build here defines it
            assert included <= set(texts), (app, language, sorted(included - set(texts)))


def test_run_args_are_the_makefile_run_target_of_every_language(pinned_root: Path) -> None:
    for app, item in suite().items.items():
        for language, spec in item.languages.items():
            makefile = (pinned_root / spec.dir / "Makefile").read_bytes().decode("utf-8")
            targets = RUN_TARGET.findall(makefile)
            assert len(targets) == 1, (app, language, targets)
            assert targets[0].split() == list(item.run_args), (app, language)


def test_passfail_languages_are_those_whose_build_files_print_pass(pinned_root: Path) -> None:
    for app, item in suite().items.items():
        printing = {
            language
            for language in item.languages
            if any(PASS_LITERAL.search(text) for text in build_files(app, language, pinned_root).values())
        }
        assert frozenset(item.passfail) == printing, app
