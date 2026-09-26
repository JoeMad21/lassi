"""Remote tests of the tt-metal host toolchain `ttmetal-host` on the build host (task P4.10).

Bible: Toolchain Pins (the joint pin; Install; the --version check against
EXPECT_VERSION in the compile sandbox before the first build), Harness
Contract (device kernels compile at first launch), Sandbox (compiles),
Agent Rules 6, 7, 9, and 10. The design is in test_ttmetal_host.py.

- build_toolchain("ttmetal-host", $LASSI_TOOLCHAINS) checks the installed
  tree $LASSI_TOOLCHAINS/tt-metal@5280a9cf against the pin (its
  lassi-install.txt names the pinned commit; its lassi-cpm-sources.txt
  equals toolchains/tt-metal-cpm-sources.txt, the list P4.2 recorded in
  results/p4-tt-install/summary.md), then runs the pinned clang++-20 with
  --version through the compile sandbox. Copies of the pin with another
  EXPECT_VERSION, and of the tracked list with one line changed, are
  refused there.
- The gate's example and the five Tier A examples build from the pinned
  sources through the toolchain's own build(), unmodified: each host
  source is read from the tree, and each kernel it names (the string after
  OVERRIDE_KERNEL_PREFIX, which the toolchain leaves undefined, so the
  example's own fallback applies; the test checks that it is "") is placed
  at that path in the build
  directory, which is the run's working directory, where the kernel JIT
  looks first. The matmul examples also get Matmul::Common's bmm_op.hpp at
  the path they include it by, since the pinned build gave them that
  directory (rx 20260925-223734-exec-4611). Each build is clean (no
  diagnostic, empty compile.stderr) and links as the pinned build linked
  the same example: readelf -d gives the same NEEDED list and a RUNPATH
  naming the same directories of the tree, so no loader variable is
  needed.
- A host program written for these tests, with a kernel beside it that
  includes a device header, builds clean (so the host compiler never built
  the kernel) and leaves the kernel at its path; an undeclared identifier
  fails the build with an error on the built file.

Nothing here runs a built program: the tests only compile and read ELF
headers with readelf. No program starts the kernel JIT or ttsim, and no
command names a device (Agent Rules 6 and 9); P4.9 and P4.11 run programs.

The tests are marked `remote` and skip unless the host can run them (Linux,
the sandbox tools and readelf on PATH, a reachable user systemd manager,
$LASSI_SCRATCH, $LASSI_RUNS_ROOT, and $LASSI_TOOLCHAINS set, TMPDIR inside
$LASSI_SCRATCH, the pin's EXECUTABLE present, and the tree installed); with
LASSI_REQUIRE_SANDBOX=1 they fail instead, so a silent skip never passes
for evidence. Run them, in the same rx run as the fixture capture, with
`uv run tools/rx.py run -- '<command>'`, the command holding
`LASSI_REQUIRE_SANDBOX=1 uv run pytest -q -m remote tests/toolchains/test_ttmetal_host_remote.py`.
Every file they write lies in a temp directory under $LASSI_RUNS_ROOT or
pytest's temp directory (under TMPDIR on the scratch disk), removed
afterwards (Agent Rule 7); the pinned tree is only read. No value here is a
measurement.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from lassi.core.runner import BuiltToolchain, RunError, build_toolchain
from lassi.executors.sandbox import SandboxedCompileRunner
from lassi.toolchains import pins as pins_module

REPO = Path(__file__).resolve().parents[2]
TOOLCHAINS_DIR = REPO / "toolchains"
CPM_LIST_NAME = "tt-metal-cpm-sources.txt"
FIXTURE_SOURCES = Path(__file__).resolve().parent / "fixtures" / "ttmetal" / "sources"
NAME = "ttmetal-host"
PIN_NAME = "tt-metal"
TOOLS = ("unshare", "systemd-run", "nice", "setpriv", "prlimit", "timeout", "python3", "awk", "readelf")
REQUIRE = os.environ.get("LASSI_REQUIRE_SANDBOX") == "1"
OUTPUT = "main"
ATTACHMENT = "compile.stderr"
# Where the examples live in the tree, each example's directory under it (rx 20260925-223734-exec-4611), and the
# directory of the header library Matmul::Common.
EXAMPLES_DIR = "tt_metal/programming_examples"
EXAMPLES = {
    "add_2_integers_in_riscv": "add_2_integers_in_riscv",
    "loopback": "loopback",
    "eltwise_binary": "eltwise_binary",
    "eltwise_sfpu": "eltwise_sfpu",
    "matmul_single_core": "matmul/matmul_single_core",
    "matmul_multi_core": "matmul/matmul_multi_core",
}
MATMUL_COMMON = "matmul/matmul_common"
# Where the pinned build put each example's binary, relative to the tree (exec 4611).
BUILT_EXAMPLES = "build_Release/programming_examples"
# A kernel path an example names: the string literal after OVERRIDE_KERNEL_PREFIX.
NAMED_KERNEL = re.compile(r'OVERRIDE_KERNEL_PREFIX\s*"([^"]+)"')
# The example's own definition when the build leaves the define unset: an empty prefix.
EMPTY_PREFIX = re.compile(r'#\s*ifndef\s+OVERRIDE_KERNEL_PREFIX\s*\n\s*#\s*define\s+OVERRIDE_KERNEL_PREFIX\s+""\s*\n')
# An include of Matmul::Common's header, in either form; the group is the path the program names.
MATMUL_INCLUDE = re.compile(r'#\s*include\s*[<"]([^>"]*bmm_op\.hpp)[>"]')
# One readelf -d entry that matters here, for example "(RUNPATH)  Library runpath: [/a:/b]".
DYNAMIC_ENTRY = re.compile(r"\((NEEDED|RUNPATH|RPATH)\)\s.*\[(.*)\]")


def host_problem() -> str:
    """Return why this host cannot run the tests, or "" when it can."""
    if not sys.platform.startswith("linux"):
        return f"the sandbox needs Linux, not {sys.platform}; run it through `uv run tools/rx.py run`"
    missing = [tool for tool in TOOLS if shutil.which(tool) is None]
    if missing:
        return f"not on PATH: {', '.join(missing)}"
    runtime = os.environ.get("XDG_RUNTIME_DIR", "")
    if not runtime or not (Path(runtime) / "bus").exists():
        return "no user systemd manager: $XDG_RUNTIME_DIR/bus does not exist"
    for name in ("LASSI_SCRATCH", "LASSI_RUNS_ROOT", "LASSI_TOOLCHAINS", "TMPDIR"):
        if not os.environ.get(name):
            return f"{name} is not set"
    scratch, tmpdir = Path(os.environ["LASSI_SCRATCH"]).resolve(), Path(os.environ["TMPDIR"]).resolve()
    if scratch not in tmpdir.parents:
        return "TMPDIR does not lie inside $LASSI_SCRATCH"
    pin = pins_module.read_pin(PIN_NAME)
    executable = pin.get("EXECUTABLE", "")
    if not executable or not Path(executable).is_file():
        return f"the pinned host compiler {executable!r} (EXECUTABLE in toolchains/tt-metal.pin) is not on this host"
    if not (installed_tree() / "lassi-install.txt").is_file():
        return f"{installed_tree()} is not an install of the pinned tt-metal"
    return ""


def installed_tree() -> Path:
    """Return the pinned tree under the resolved toolchains root."""
    return Path(os.environ["LASSI_TOOLCHAINS"]).resolve() / pins_module.read_pin(PIN_NAME)["PREFIX_NAME"]


PROBLEM = host_problem()
pytestmark = [
    pytest.mark.remote,
    pytest.mark.skipif(bool(PROBLEM) and not REQUIRE, reason=PROBLEM or "the host can build tt-metal programs"),
]


@pytest.fixture(autouse=True)
def host_ready() -> None:
    """Fail, rather than skip, when LASSI_REQUIRE_SANDBOX=1 asks for a real sandbox and the host cannot run one."""
    if PROBLEM:
        pytest.fail(f"LASSI_REQUIRE_SANDBOX=1, but {PROBLEM}")


@pytest.fixture(scope="module")
def built() -> BuiltToolchain:
    """Build ttmetal-host as a run builds it: the tree checked, the pinned clang checked by --version."""
    if PROBLEM:
        pytest.fail(f"LASSI_REQUIRE_SANDBOX=1, but {PROBLEM}")
    return build_toolchain(NAME, Path(os.environ["LASSI_TOOLCHAINS"]))


@pytest.fixture
def workdir() -> Iterator[Path]:
    """Create a fresh attempt build directory under $LASSI_RUNS_ROOT; remove it afterwards."""
    base = Path(tempfile.mkdtemp(prefix="lassi-ttmetal-host.", dir=os.environ["LASSI_RUNS_ROOT"]))
    try:
        build = base / "attempt00" / "build"
        build.mkdir(parents=True)
        yield build
    finally:
        shutil.rmtree(base)


def text_of(path: Path) -> str:
    """Return a file of the pinned tree as text, decoded as UTF-8."""
    return path.read_bytes().decode("utf-8")


def dynamic(path: Path) -> dict[str, list[str]]:
    """Return the NEEDED, RUNPATH, and RPATH entries readelf -d reads from an ELF file; nothing runs it."""
    environment = {"PATH": os.environ["PATH"], "LANG": "C", "LC_ALL": "C"}
    done = subprocess.run(
        ["readelf", "-d", str(path)], capture_output=True, text=True, env=environment, timeout=60, check=False
    )
    assert done.returncode == 0, done.stderr
    entries: dict[str, list[str]] = {"NEEDED": [], "RUNPATH": [], "RPATH": []}
    for line in done.stdout.splitlines():
        match = DYNAMIC_ENTRY.search(line)
        if match:
            entries[match[1]].append(match[2])
    return entries


def runpath_dirs(entries: dict[str, list[str]]) -> list[Path]:
    """Return the resolved directories of the one RUNPATH entry; fail when there is not exactly one."""
    assert len(entries["RUNPATH"]) == 1 and not entries["RPATH"], entries
    return [Path(directory).resolve() for directory in entries["RUNPATH"][0].split(":")]


def cpm_lines(text: str) -> list[str]:
    """Return the non-comment, non-blank lines of a CPM sources list, trailing whitespace removed."""
    return [line.rstrip() for line in text.splitlines() if line.strip() and not line.startswith("#")]


def copy_pins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Copy every file of toolchains/ into a temp pins directory and point the pin reader at it."""
    pins_dir = tmp_path / "pins"
    pins_dir.mkdir()
    for path in TOOLCHAINS_DIR.iterdir():
        if path.is_file():
            shutil.copyfile(path, pins_dir / path.name)
    monkeypatch.setattr(pins_module, "PINS_DIR", pins_dir)
    return pins_dir


def test_the_version_check_ran_the_pinned_clang_in_the_compile_sandbox(built: BuiltToolchain) -> None:
    pin = pins_module.read_pin(PIN_NAME)
    assert built.executable == pin["EXECUTABLE"]
    assert built.version_status == 0
    assert pin["EXPECT_VERSION"] in "\n".join(built.version)
    assert built.pins == {PIN_NAME: pin}
    assert isinstance(built.toolchain.runner, SandboxedCompileRunner), "every compile runs in the compile sandbox"
    assert built.environment is not None and set(built.environment) == {"PATH", "LANG", "LC_ALL"}


def test_the_toolchain_builds_against_the_installed_tree(built: BuiltToolchain) -> None:
    assert Path(built.toolchain.tree) == installed_tree()


def test_the_installed_tree_is_the_pinned_install_with_the_recorded_cpm_sources() -> None:
    tree = installed_tree()
    first = (tree / "lassi-install.txt").read_text(encoding="ascii").splitlines()[0].split()
    assert first[:2] == [PIN_NAME, pins_module.read_pin(PIN_NAME)["COMMIT"]]
    tracked = cpm_lines((TOOLCHAINS_DIR / CPM_LIST_NAME).read_text(encoding="ascii"))
    assert cpm_lines((tree / "lassi-cpm-sources.txt").read_text(encoding="ascii")) == tracked


@pytest.mark.parametrize("example", sorted(EXAMPLES))
def test_an_upstream_example_builds_from_the_pinned_sources(built: BuiltToolchain, workdir: Path, example: str) -> None:
    tree = installed_tree()
    examples = tree / EXAMPLES_DIR
    source = text_of(examples / EXAMPLES[example] / f"{example}.cpp")
    assert EMPTY_PREFIX.search(source), f"{example} names kernels relative to its working directory when unset"
    named = NAMED_KERNEL.findall(source)
    assert named, f"{example} names its kernels after OVERRIDE_KERNEL_PREFIX"
    files = {f"{example}.cpp": source, **{path: text_of(examples / path) for path in named}}
    harness = {path: text_of(examples / MATMUL_COMMON / "bmm_op.hpp") for path in MATMUL_INCLUDE.findall(source)}
    result = built.toolchain.build(files, workdir, harness=harness)
    stderr = (workdir / ATTACHMENT).read_text(encoding="utf-8")
    assert result.artifact == workdir / OUTPUT, f"{example} did not build:\n{stderr}"
    assert result.diagnostics == [] and stderr == "", f"a clean build prints nothing:\n{stderr}"
    for path in named:
        placed = result.artifact.parent / path
        assert placed.read_bytes() == (examples / path).read_bytes(), f"{path} lies where the kernel JIT looks first"
    ours, pinned = dynamic(result.artifact), dynamic(tree / BUILT_EXAMPLES / f"metal_example_{example}")
    assert ours["NEEDED"] == pinned["NEEDED"], "linked against the same libraries as the pinned build"
    assert runpath_dirs(ours) == runpath_dirs(pinned), "the same RUNPATH, so no loader variable is needed"
    assert all(tree in directory.parents for directory in runpath_dirs(ours)), "every RUNPATH lies in the tree"


def test_an_own_host_program_builds_and_its_kernel_is_placed_not_compiled(built: BuiltToolchain, workdir: Path) -> None:
    case = FIXTURE_SOURCES / "ttm_clean"
    files = {path.relative_to(case).as_posix(): text_of(path) for path in sorted(case.rglob("*")) if path.is_file()}
    kernels = [path for path in files if path.startswith("kernels/")]
    assert kernels
    result = built.toolchain.build(files, workdir)
    stderr = (workdir / ATTACHMENT).read_text(encoding="utf-8")
    assert result.artifact == workdir / OUTPUT, f"the program did not build:\n{stderr}"
    assert result.diagnostics == [] and stderr == "", "a kernel given to the host compiler would fail on its include"
    for path in kernels:
        assert (workdir / path).read_text(encoding="utf-8") == files[path]
    entries = dynamic(result.artifact)
    assert "libtt_metal.so" in entries["NEEDED"]
    assert all(installed_tree() in directory.parents for directory in runpath_dirs(entries))


def test_a_compile_error_parses_into_an_error_on_the_built_file(built: BuiltToolchain, workdir: Path) -> None:
    source = text_of(FIXTURE_SOURCES / "ttm_undeclared_identifier" / "main.cpp")
    result = built.toolchain.build({"main.cpp": source}, workdir)
    assert result.artifact is None
    errors = [item for item in result.diagnostics if item.severity == "error"]
    assert any(item.file == "main.cpp" and item.line is not None for item in errors), result.diagnostics
    assert (workdir / ATTACHMENT).read_text(encoding="utf-8").strip(), "the raw stderr is kept"


def test_a_clang_whose_version_is_not_the_pin_is_refused_in_the_sandbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pins_dir = copy_pins(tmp_path, monkeypatch)
    text = (pins_dir / "tt-metal.pin").read_text(encoding="ascii")
    expected = pins_module.read_pin(PIN_NAME)["EXPECT_VERSION"]
    changed = text.replace(expected, "PLACEHOLDER version text that no compiler prints")
    assert changed != text
    (pins_dir / "tt-metal.pin").write_text(changed, encoding="ascii", newline="\n")
    with pytest.raises(RunError, match="EXPECT_VERSION"):
        build_toolchain(NAME, Path(os.environ["LASSI_TOOLCHAINS"]))


def test_cpm_sources_that_differ_from_the_tracked_list_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pins_dir = copy_pins(tmp_path, monkeypatch)
    path = pins_dir / CPM_LIST_NAME
    lines = path.read_text(encoding="ascii").splitlines()
    index = next(number for number, line in enumerate(lines) if line.strip() and not line.startswith("#"))
    lines[index] = lines[index] + " PLACEHOLDER-drift"
    path.write_text("".join(f"{line}\n" for line in lines), encoding="ascii", newline="\n")
    with pytest.raises(RunError, match="lassi-cpm-sources.txt"):
        build_toolchain(NAME, Path(os.environ["LASSI_TOOLCHAINS"]))
