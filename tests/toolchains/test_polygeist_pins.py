"""Tests for the Polygeist pin file, its tracked list, and its install script (task P5.5).

Bible: Toolchain Pins (MLIR stacks; pin files and install rules), Language
Frontends (C and C++ rows), IR Strategy (cpu-mlir and llvm-ir rows), Result
Record (Trial.toolchain_pins), Host Facts (the 150G scratch cap and the 145G
stop line), Agent Rules 7 and 10; plans/p5-ir.md, P5.5;
plans/spikes/p5-mlir-pins.md, Results 1, 4, and 5 and "Consequences for the
plan" (the P5.5 bullet is the design).

The design: toolchains/polygeist.sh builds LLVM 26eb4285 (the llvm-project
submodule commit of Polygeist 77c04bb2) with clang and mlir, and Polygeist
as LLVM's external project, in one in-tree build (Polygeist README.md:89-95),
for the host target, with Polymer, CUDA, and ROCm off, LLVM's tests off,
POLYGEIST_PGO_DEFAULT_DATA_DIR under the scratch root, CMake named by path,
and the host's clang-17 as compiler. It fetches Polygeist and llvm-project by
commit at depth 1, never through `git submodule`, and checks that
Polygeist's tree records LLVM_COMMIT as its llvm-project gitlink. It builds
only TARGETS (cgeist, polygeist-opt, and that LLVM's mlir-opt,
mlir-translate, and clang, and LLVMFrontendOpenACC for P12's flang), in place, with no install, and checks each
tool's --version at the end. cgeist stops at MLIR in P5's use (-S), so its
/tmp file (tools/cgeist/driver.cc:1200-1202) is never written; the remote
tests (test_polygeist_remote.py) run it so.

Decision on the order with P5.4 (point 3 of the brief, revised 2026-10-26 so
the build can run on alpha01 before tt-mlir's): the script does not wait for
tt-mlir's install, and its space check plans for both MLIR stacks together
whichever is built first. When tt-mlir's install is complete (the tt-mlir
prefix holds lassi-install.txt whose first line names tt-mlir and its
pinned COMMIT, and no lassi-install.unfinished), `du -s` of the scratch root
already counts that stack, so the plan is INSTALL_KIB alone, the Polygeist
stack's projected upper end, which equals the RESERVE_KIB tt-mlir.pin holds
for it. Otherwise the plan adds RESERVE_KIB, the tt-mlir stack, which equals
tt-mlir.pin's INSTALL_KIB. That check only reads, and it runs before the
space check and before any download. The big jobs still run one at a time
(the gate's limit on alpha01, the long-job runner's lock on the
workstation).

Decision on Trial.toolchain_pins (point 4): no schema change. llvm holds the
CPU stack's LLVM (26eb4285), the one whose own tools (mlir-opt,
mlir-translate, clang) a trial runs; polygeist holds polygeist.pin's
VERSION; tt_mlir holds tt-mlir.pin's VERSION, whose pin names its LLVM
4efe170d, which is never recorded in the llvm field. The value the llvm
field records is polygeist.pin's LLVM_VERSION.

The contract these tests fix, for the implementation to follow:

- toolchains/polygeist.pin keys: NAME, VERSION, PREFIX_NAME, COMMIT, URL;
  LLVM_COMMIT, LLVM_URL, and LLVM_SUBMODULE (the gitlink's path in
  Polygeist's tree); LLVM_VERSION (what Trial.toolchain_pins.llvm records);
  LLVM_CMAKE_FLAGS ("-G Ninja" then -DKEY=VALUE words, no path); TARGETS;
  the host tools CMAKE, NINJA, CLANG_C, CLANG_CXX, and PYTHON, each with its
  *_EXPECT version text; CGEIST, POLYGEIST_OPT, MLIR_OPT, MLIR_TRANSLATE,
  and LLVM_CLANG (paths relative to the prefix, in one bin directory);
  EXPECT_VERSION (what cgeist, polygeist-opt, mlir-opt, and mlir-translate
  print) and LLVM_CLANG_EXPECT (what that clang prints); INSTALL_KIB (the
  Polygeist stack) and RESERVE_KIB (the tt-mlir stack).
- toolchains/polygeist-fetched-sources.txt: the tracked list of what the
  configure fetched, PLACEHOLDER until the first install records it, the
  one line none, or lines in toolchains/tt-metal-cpm-sources.txt's forms.
- toolchains/polygeist.sh, in tt-mlir.sh's style (`name() {` functions, the
  main steps after `trap cleanup EXIT`, the record lassi-install.txt and the
  marker lassi-install.unfinished, messages starting "polygeist:"):
  pinned_record, check_prefix, tt_mlir_installed, expect_version,
  check_host, check_space <KiB>, check_gitlink <repository dir>,
  llvm_settings (the -D settings the script adds to the one configure),
  check_cache <CMakeCache.txt> <flags>, record_fetched_sources,
  check_fetched_sources <tree list> <tracked list>, verify_install, and
  check_root_fs. A kept install's message says "already holds the pinned";
  a refused prefix's names PREFIX_NAME and says agents "never delete" there;
  a refusal of check_gitlink names LLVM_COMMIT, and tt_mlir_installed's
  messages name tt-mlir. The probes below run them in bash with the pin's
  values sourced and here (holding polygeist.pin, its tracked list, and
  tt-mlir.pin), prefix, record_name, unfinished_name, LASSI_ALLOWED_ROOT,
  LASSI_STOP_GIB, LASSI_TOOLCHAINS, LASSI_SCRATCH, LASSI_JOBS, HOME, and
  TMPDIR set, plus the script's top-level lowercase assignments of paths
  under $prefix or $LASSI_SCRATCH (its source, build, and PGO directories);
  pinned_record runs no host tool.
- The host values (task P5.15, as toolchains/tt-mlir.sh takes them): the
  script takes its allowed root from LASSI_ALLOWED_ROOT and its stop line
  from LASSI_STOP_GIB (a positive integer of GiB), both required with no
  default and checked with the other values before the core limit. The
  host profile (tools/local/host-profile.sh) sets them on the workstation;
  on alpha01 the rx command line gives
  LASSI_ALLOWED_ROOT=/mnt/nvme10/joseph_ufl and LASSI_STOP_GIB=145.
  LASSI_TOOLCHAINS, LASSI_SCRATCH, and HOME must each resolve to the allowed
  root or under it, which is matched literally, and a relative allowed root
  or "/" is refused. The whole-script probes run the script with stand-ins
  for prlimit and for every tool that writes or fetches, each an exported
  bash function that logs its call and fails, so the script stops at its
  first step past its checks and writes nothing; the stop line values other
  than 145 are SYNTHETIC.

No test fetches from the network, configures, or builds anything, and no
test runs a program that could open a device. Fixture texts are SYNTHETIC.
No value here is a measurement: commits, options, and line numbers come from
upstream source read with `gh api` on 2026-10-13 (file:line at the commit
named); host tool versions from the P5.1 spike's read-only probes (rx ids
cited); space figures are the spike's PROJECTED upper ends.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from lassi.core.record import TOOLCHAIN_PIN_NAMES, ToolchainPins
from lassi.toolchains.pins import prefix_pin_name, read_pin

REPO = Path(__file__).resolve().parents[2]
TOOLCHAINS = REPO / "toolchains"
BIBLE = REPO / "docs" / "BIBLE.md"
GATE_CONFIG = REPO / "tools" / "server" / "config.default.json"
PIN_NAME = "polygeist"
TT_MLIR_PIN_NAME = "tt-mlir"
SCRIPT_NAME = "polygeist.sh"
TRACKED_FETCHED_NAME = "polygeist-fetched-sources.txt"
TREE_FETCHED_NAME = "lassi-fetched-sources.txt"
RECORD_NAME = "lassi-install.txt"
UNFINISHED_NAME = "lassi-install.unfinished"
# The bash on PATH, resolved here: on Windows a bare "bash" given to subprocess can start the WSL launcher in
# System32 instead, which sees neither this environment nor Windows paths.
BASH = shutil.which("bash")
needs_bash = pytest.mark.skipif(BASH is None, reason="bash is not available")

# Upstream values, read with `gh api` on 2026-10-13.
# llvm/Polygeist main's head, 2024-07-31T23:43:12Z, "Implement builtin functions __builtin_ctz{s,,l,ll} (#424)" (the
# commits API); bible Toolchain Pins, MLIR stacks. Polygeist has no release and one tag, cgo24 (the releases and tags
# APIs), so VERSION is the commit's first eight characters, as tt-metal.pin's is. README.md:10 clones the repository
# at https://github.com/llvm/Polygeist; the URL takes tt-mlir.pin's .git form.
POLYGEIST_COMMIT = "77c04bb2a7a2406ca9480bcc9e729b07d2c8d077"
POLYGEIST_URL = "https://github.com/llvm/Polygeist.git"
# The llvm-project gitlink at POLYGEIST_COMMIT (contents/llvm-project: type submodule, this sha, and
# submodule_git_url https://github.com/llvm/llvm-project.git), as .gitmodules:1-3 names it (path llvm-project).
LLVM_COMMIT = "26eb4285b56edd8c897642078d91f16ff0fd3472"
LLVM_URL = "https://github.com/llvm/llvm-project.git"
LLVM_SUBMODULE = "llvm-project"
# tt-mlir's LLVM (toolchains/tt-mlir.pin, LLVM_COMMIT), which never enters this stack or the llvm field.
TT_LLVM_COMMIT = "4efe170d858eb54432f520abb4e7f0086236748b"
# What cgeist, mlir-opt, and mlir-translate print for --version: LLVM's printer, "<PACKAGE_NAME> version
# <PACKAGE_VERSION>" (llvm/lib/Support/CommandLine.cpp:2545-2556 at LLVM_COMMIT), PACKAGE_NAME LLVM
# (llvm/CMakeLists.txt:336) and PACKAGE_VERSION 18.0.0git (llvm/CMakeLists.txt:11-26). cgeist parses its options
# with cl::ParseCommandLineOptions (tools/cgeist/driver.cc:521 at POLYGEIST_COMMIT) and registers no printer.
LLVM_VERSION_LINE = "LLVM version 18.0.0git"
# What that clang prints: "clang version " CLANG_VERSION_STRING (clang/lib/Basic/Version.cpp:92-105;
# clang/include/clang/Basic/Version.inc.in:2), where CLANG_VERSION leaves the "git" suffix out
# (clang/CMakeLists.txt:293-294), then "(<origin URL> <HEAD>)" when the source is a git checkout
# (llvm/cmake/modules/VersionFromVCS.cmake:17-46; LLVM_APPEND_VC_REV is ON, llvm/CMakeLists.txt:325-326).
CLANG_VERSION = re.compile(r"clang version 18\.0\.0(?: \((\S+) ([0-9a-f]{40})\))?")
# Options Polygeist's CMakeLists.txt defines at POLYGEIST_COMMIT, by line (POLYGEIST_USE_LINKER is read at :90 and
# defined nowhere). A POLYGEIST_ or POLYMER_ name outside these is a typo CMake would ignore.
POLYGEIST_OPTIONS = {
    "POLYGEIST_ENABLE_CUDA": 5,
    "POLYGEIST_ENABLE_ROCM": 6,
    "POLYGEIST_ENABLE_POLYMER": 8,
    "POLYGEIST_POLYMER_ENABLE_ISL": 9,
    "POLYGEIST_POLYMER_ENABLE_PLUTO": 10,
    "POLYMER_DEP_DIR": 12,
    "POLYGEIST_PGO_DEFAULT_DATA_DIR": 44,
    "POLYGEIST_PGO_ALTERNATIVE_ENV_VAR": 45,
    "POLYGEIST_PGO_DATA_DIR_ENV_VAR": 46,
    "POLYGEIST_USE_LINKER": 90,
}
# Off, and set so that check_cache reads them back: CUDA (:5), ROCm (:6), and Polymer (:8), each 0 by default.
OFF_OPTIONS = ("POLYGEIST_ENABLE_CUDA", "POLYGEIST_ENABLE_ROCM", "POLYGEIST_ENABLE_POLYMER")
# The tools P5 calls, by target: cgeist (tools/cgeist/CMakeLists.txt:20, add_clang_executable), polygeist-opt
# (tools/polygeist-opt/CMakeLists.txt:18), which runs Polygeist's passes on what cgeist writes, mlir-opt and
# mlir-translate (mlir/tools/mlir-opt/CMakeLists.txt:82, mlir/tools/mlir-translate/CMakeLists.txt:8, add_mlir_tool),
# and clang (clang/tools/driver/CMakeLists.txt:26), which depends on clang-resource-headers (:59), the builtin
# headers cgeist finds from its own path (tools/cgeist/Lib/clang-mlir.cc:6027-6031). clang-resource-headers
# (clang/lib/Headers/CMakeLists.txt:434) may be added; nothing else. LLVMFrontendOpenACC, whose acc_gen step writes
# the OpenACC headers flang's parser includes, is built for P12's flang (OQ-063 (a), answered 2026-10-26).
REQUIRED_TARGETS = {"cgeist", "polygeist-opt", "mlir-opt", "mlir-translate", "clang", "LLVMFrontendOpenACC"}
ALLOWED_TARGETS = REQUIRED_TARGETS | {"clang-resource-headers"}
# The built tools, by pin key and executable name. In the in-tree build every tool goes to the build tree's bin
# (Polygeist CMakeLists.txt:38 sets LLVM_RUNTIME_OUTPUT_INTDIR to ${CMAKE_BINARY_DIR}/bin), so cgeist finds clang's
# resource directory, <bin>/../lib/clang/18 (cmake/Modules/GetClangResourceDir.cmake:18 at LLVM_COMMIT).
TOOLS = {
    "CGEIST": "cgeist",
    "POLYGEIST_OPT": "polygeist-opt",
    "MLIR_OPT": "mlir-opt",
    "MLIR_TRANSLATE": "mlir-translate",
    "LLVM_CLANG": "clang",
}
# Host tools on alpha01 (plans/spikes/p5-mlir-pins.md, Results 4; rx 20261006-214920-exec-4a93): CMake by path, Ninja,
# clang-17 (the compiler toolchains/tt-mlir.pin names; Polygeist's CI builds LLVM and Polygeist with clang,
# .github/workflows/build.yml:23-24, :37-38, :91, :107), and Python 3.10, which reads the core limit back.
CMAKE_CHOICES = {("/usr/local/bin/cmake", "cmake version 3.31.6"), ("/usr/bin/cmake", "cmake version 4.2.1")}
# Space, PROJECTED (plans/spikes/p5-mlir-pins.md, Results 5, configuration A, upper ends, in hundredths of a GiB):
# the Polygeist stack, the tt-mlir stack, and both stacks together, which the bible's MLIR stacks bullet holds under
# the 145G stop line.
GIB_KIB = 1024 * 1024
POLYGEIST_STACK_HUNDREDTHS = 1915
TT_MLIR_STACK_HUNDREDTHS = 2330
BOTH_STACKS_HUNDREDTHS = 4245
STOP_KIB = 145 * GIB_KIB
# SYNTHETIC values for the probes' mismatches.
SYNTHETIC_COMMIT = "0123456789abcdef0123456789abcdef01234567"
IDENTITY = {
    "GIT_AUTHOR_NAME": "Pin Test",
    "GIT_AUTHOR_EMAIL": "pin-test@example.invalid",
    "GIT_COMMITTER_NAME": "Pin Test",
    "GIT_COMMITTER_EMAIL": "pin-test@example.invalid",
}

FUNCTION = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\(\) \{\n.*?^\}\n", re.S | re.M)
SETTING = re.compile(r"-D([A-Za-z0-9_]+)(?::[A-Z]+)?=(\S+)")
# A command that downloads: git fetch, clone, pull, or submodule (directly or through a git_in wrapper), pip install
# or download, curl, or wget.
DOWNLOAD = re.compile(
    r"\bgit(?:_in)?\b.*\b(?:fetch|clone|pull|submodule)\b|\bpip3?\b.*\b(?:install|download)\b|\bcurl\b|\bwget\b"
)
# One entry of a recorded fetched-sources list, in toolchains/tt-metal-cpm-sources.txt's two forms.
FETCHED_ENTRY = re.compile(r"\S+ (?:git [0-9a-f]{40} changed=\d+|files-sha256 [0-9a-f]{64})")


# Helpers.


def pin() -> dict[str, str]:
    """Return toolchains/polygeist.pin as the repository's pin reader reads it."""
    return read_pin(PIN_NAME)


def script_text() -> str:
    """Return the text of toolchains/polygeist.sh."""
    return (TOOLCHAINS / SCRIPT_NAME).read_text(encoding="utf-8")


def kib_of(hundredths: int) -> int:
    """Return hundredths of a GiB in KiB, rounded up."""
    return -(-hundredths * GIB_KIB // 100)


def code_lines(text: str) -> list[str]:
    """Return the stripped code lines of bash text, continuations joined, without comments or blank lines."""
    joined = re.sub(r"\\\n\s*", " ", text)
    lines = [line.strip() for line in joined.splitlines()]
    return [line for line in lines if line and not line.startswith("#")]


def logic_lines(text: str) -> list[str]:
    """Return the code lines of bash text that are not echo lines (messages, and settings a function prints)."""
    return [line for line in code_lines(text) if not line.startswith("echo ")]


def functions(text: str) -> dict[str, str]:
    """Return every top-level bash function of a script, its name mapped to its definition through its `}` line."""
    return {match.group(1): match.group(0) for match in FUNCTION.finditer(text)}


def bash_function(text: str, name: str) -> str:
    """Return the definition of the top-level bash function `name`; fail when the script has none."""
    found = functions(text)
    assert name in found, f"the script defines no function {name}"
    return found[name]


def body_lines(text: str, name: str) -> list[str]:
    """Return the logic lines of the function `name` after its `name() {` line."""
    return logic_lines(bash_function(text, name))[1:]


def main_section(text: str) -> str:
    """Return the part of a script after its cleanup trap is set: the install steps, without the functions."""
    return text[text.index("\ntrap cleanup EXIT\n") :]


def calls(line: str, names: set[str]) -> bool:
    """Return whether a code line runs one of the named functions as a command word."""
    return any(re.search(rf"(^|[\s;&|(!]){re.escape(name)}($|[\s;)])", line) for name in names)


def positioned(main: str) -> list[tuple[int, str]]:
    """Return (offset, stripped line) for each code line of `main` that is not a comment, blank, or echo line."""
    found, offset = [], 0
    for raw in main.splitlines(keepends=True):
        line = raw.strip()
        if line and not line.startswith("#") and not line.startswith("echo ") and not line.startswith('"'):
            found.append((offset, line))
        offset += len(raw)
    return found


def call_position(main: str, names: set[str], after: int = -1) -> int:
    """Return the offset in `main` of the first code line past `after` that runs one of `names`; fail when none does."""
    for offset, line in positioned(main):
        if offset > after and calls(line, names):
            return offset
    raise AssertionError(f"the main steps never run {sorted(names)}")


def step_position(text: str, marker: str, after: int) -> int:
    """Return the main-section offset past `after` of the first line with `marker` or a call of a function with it."""
    main = main_section(text)
    found = [offset for offset, line in positioned(main) if offset > after and marker in line]
    holders = {name for name, body in functions(text).items() if marker in body}
    found += [offset for offset, line in positioned(main) if offset > after and calls(line, holders)]
    assert found, f"no main step after offset {after} holds {marker}"
    return min(found)


def downloading_functions(text: str) -> set[str]:
    """Return the functions that download, directly or through another such function."""
    bodies = {name: body_lines(text, name) for name in functions(text)}
    found = {name for name, lines in bodies.items() if any(DOWNLOAD.search(line) for line in lines)}
    grown = True
    while grown:
        more = {name for name, lines in bodies.items() if name not in found and any(calls(ln, found) for ln in lines)}
        found |= more
        grown = bool(more)
    return found


def first_download(text: str) -> int:
    """Return the offset in the main section of the first step that downloads; fail when none does."""
    main, downloaders = main_section(text), downloading_functions(text)
    for offset, line in positioned(main):
        if DOWNLOAD.search(line) or calls(line, downloaders):
            return offset
    raise AssertionError("the main steps download nothing; the pinned sources must be fetched")


def builders(text: str) -> set[str]:
    """Return the functions that run a CMake build of the pin's TARGETS."""
    return {name for name, body in functions(text).items() if "--build" in body and "$TARGETS" in body}


def cmake_settings(flags: str) -> dict[str, str]:
    """Return the -DKEY=VALUE settings of a flags value; fail on anything but those and a leading -G Ninja."""
    words = flags.split()
    assert words[:2] == ["-G", "Ninja"], "the Ninja generator comes first"
    settings: dict[str, str] = {}
    for word in words[2:]:
        match = SETTING.fullmatch(word)
        assert match, f"not a -DKEY=VALUE setting: {word}"
        assert match.group(1) not in settings, f"{match.group(1)} is set twice"
        settings[match.group(1)] = match.group(2)
    return settings


def entry_lines(path: Path) -> list[str]:
    """Return the lines of a tracked list that are not blank and not # comments, trailing whitespace removed."""
    text = path.read_bytes().decode("ascii")
    return [line.rstrip() for line in text.splitlines() if line.strip() and not line.startswith("#")]


def section(text: str, heading: str) -> str:
    """Return the body of the markdown section `heading` up to the next heading of any level."""
    start = text.index(f"\n{heading}\n")
    end = text.find("\n#", start + len(heading) + 2)
    return text[start : end if end != -1 else len(text)]


def prefix_state(prefix: Path) -> dict[str, bytes]:
    """Return every file under `prefix` by relative path, with its bytes ({} when the prefix does not exist)."""
    if not prefix.exists():
        return {}
    return {path.relative_to(prefix).as_posix(): path.read_bytes() for path in prefix.rglob("*") if path.is_file()}


def run_probe(body: str, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Write `body` to probe.sh in `cwd` and run it in bash with the test git identity."""
    (cwd / "probe.sh").write_bytes(body.encode("ascii"))
    assert BASH is not None
    return subprocess.run(
        [BASH, "probe.sh"], cwd=cwd, capture_output=True, text=True, env={**os.environ, **IDENTITY, **(env or {})}
    )


def run_functions(
    names: tuple[str, ...], body: str, cwd: Path, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run `body` in strict bash in `cwd`, after the named functions of toolchains/polygeist.sh."""
    text = script_text()
    return run_probe("set -euo pipefail\n" + "".join(bash_function(text, name) for name in names) + body, cwd, env)


# The globals the whole-script probes set, after the pin's own values (module docstring).
GLOBALS = """set -euo pipefail
here="$PWD/pins"
source "$here/polygeist.pin"
LASSI_ALLOWED_ROOT="$PWD"
LASSI_STOP_GIB=145
LASSI_TOOLCHAINS="$PWD/tc"
LASSI_SCRATCH="$PWD"
LASSI_JOBS=2
HOME="$PWD"
TMPDIR="$PWD/tmp"
prefix="$LASSI_TOOLCHAINS/$PREFIX_NAME"
record_name=lassi-install.txt
unfinished_name=lassi-install.unfinished
"""
# The script's top-level path variables (module docstring): lowercase names assigned a path under $prefix or
# $LASSI_SCRATCH, such as its source, build, and PGO directories.
LAYOUT = re.compile(r'^[a-z][a-z0-9_]*="\$(?:prefix|LASSI_SCRATCH)/[^"\n]*"$', re.M)


def run_with_script(body: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run `body` after GLOBALS, the script's layout lines, and its functions, with the pin files copied to pins/."""
    (cwd / "pins").mkdir(exist_ok=True)
    for name in ("polygeist.pin", TRACKED_FETCHED_NAME, "tt-mlir.pin"):
        shutil.copyfile(TOOLCHAINS / name, cwd / "pins" / name)
    text = script_text()
    layout = LAYOUT.findall(text)
    assert len(layout) >= 3, f"the script names its two sources and its build tree under the prefix: {layout}"
    return run_probe(GLOBALS + "".join(f"{line}\n" for line in layout) + "".join(functions(text).values()) + body, cwd)


# The pin file.


def test_pin_parses_and_names_its_prefix() -> None:
    """The pin reads through lassi.toolchains.pins and names its install prefix <NAME>@<VERSION>."""
    values = pin()
    assert values["NAME"] == PIN_NAME
    assert values["VERSION"] == POLYGEIST_COMMIT[:8], "no release names the commit; its prefix does (tt-metal.pin)"
    assert values["PREFIX_NAME"] == f"{PIN_NAME}@{values['VERSION']}"
    assert prefix_pin_name(values["PREFIX_NAME"]) == PIN_NAME


def test_pin_holds_the_mlir_stacks_polygeist_commit() -> None:
    """COMMIT is the commit the P5 plan's rule chose (bible Toolchain Pins, MLIR stacks), from llvm/Polygeist."""
    values = pin()
    assert (values["COMMIT"], values["URL"]) == (POLYGEIST_COMMIT, POLYGEIST_URL)


def test_pin_holds_the_llvm_submodule_commit() -> None:
    """LLVM is Polygeist's own submodule commit, fetched by commit from the URL .gitmodules names."""
    values = pin()
    assert (values["LLVM_COMMIT"], values["LLVM_URL"]) == (LLVM_COMMIT, LLVM_URL)
    assert values["LLVM_SUBMODULE"] == LLVM_SUBMODULE, ".gitmodules:2 path"
    assert TT_LLVM_COMMIT not in "\n".join(values.values()), "tt-mlir's LLVM is not this stack's (two stacks)"


def test_pin_builds_llvm_clang_mlir_and_polygeist_in_one_tree() -> None:
    """README.md:89-95's in-tree build, for the host target, with LLVM's tests off and its utils and tools kept."""
    settings = cmake_settings(pin()["LLVM_CMAKE_FLAGS"])
    assert set(settings["LLVM_ENABLE_PROJECTS"].split(";")) == {"clang", "mlir"}, "README.md:90; no polly or lld"
    assert settings["LLVM_EXTERNAL_PROJECTS"] == "polygeist", "README.md:91; llvm/tools/CMakeLists.txt:54-57"
    # cgeist links ${LLVM_TARGETS_TO_BUILD} (tools/cgeist/CMakeLists.txt:2); LLVM's default is all
    # (llvm/CMakeLists.txt:469).
    assert settings["LLVM_TARGETS_TO_BUILD"] in ("host", "X86"), "README.md:93"
    assert settings["CMAKE_BUILD_TYPE"] in ("Release", "MinSizeRel"), "the P5 plan's Release (README.md:95: DEBUG)"
    assert settings["LLVM_ENABLE_ASSERTIONS"] in ("ON", "OFF"), "recorded either way (README.md:94 has ON)"
    # MLIR's and clang's tests follow LLVM_INCLUDE_TESTS (mlir/CMakeLists.txt:126-128, clang/CMakeLists.txt:438-440),
    # and mlir-opt links MLIR's test libraries only with them (mlir/tools/mlir-opt/CMakeLists.txt:14).
    assert settings["LLVM_INCLUDE_TESTS"] == "OFF", "LLVM's default is ON (llvm/CMakeLists.txt:737)"
    # Polygeist's lit targets still name FileCheck, not, count, split-file, and llvm-config as dependencies
    # (test/CMakeLists.txt:8-12; tools/cgeist/Test/CMakeLists.txt:11-22), which a configure needs to exist: they
    # come from LLVM_INCLUDE_UTILS (llvm/CMakeLists.txt:717, :1168-1175) and LLVM_INCLUDE_TOOLS, so both stay on.
    assert settings.get("LLVM_INCLUDE_UTILS", "ON") == "ON"
    assert settings.get("LLVM_INCLUDE_TOOLS", "ON") == "ON"
    assert settings.get("MLIR_ENABLE_BINDINGS_PYTHON", "OFF") == "OFF", "no bindings (the P5 plan)"
    assert "LLVM_ENABLE_RUNTIMES" not in settings


def test_pin_turns_off_polymer_cuda_and_rocm() -> None:
    """CUDA, ROCm, and Polymer are set off (CMakeLists.txt:5-10), so the cache records them; ISL and Pluto if set."""
    settings = cmake_settings(pin()["LLVM_CMAKE_FLAGS"])
    for name in OFF_OPTIONS:
        assert settings.get(name) in ("0", "OFF"), f"{name} (CMakeLists.txt:{POLYGEIST_OPTIONS[name]}) must be set off"
    for name in ("POLYGEIST_POLYMER_ENABLE_ISL", "POLYGEIST_POLYMER_ENABLE_PLUTO"):
        assert settings.get(name, "0") in ("0", "OFF"), name
    for name in ("MLIR_ENABLE_CUDA_RUNNER", "MLIR_ENABLE_ROCM_RUNNER"):
        assert settings.get(name, "0") in ("0", "OFF"), f"{name} is on (README.md:33-35)"


def test_pin_names_only_options_polygeist_defines() -> None:
    """A misspelled Polygeist option would be ignored by CMake; every POLYGEIST_ or POLYMER_ name is one it defines."""
    settings = cmake_settings(pin()["LLVM_CMAKE_FLAGS"])
    ours = {name for name in settings if name.startswith(("POLYGEIST_", "POLYMER_"))}
    assert ours <= set(POLYGEIST_OPTIONS), sorted(ours - set(POLYGEIST_OPTIONS))


def test_pin_flags_name_no_path_and_only_the_pinned_compilers() -> None:
    """Paths come from the script (llvm_settings); a compiler the flags name is the pinned one."""
    values = pin()
    settings = cmake_settings(values["LLVM_CMAKE_FLAGS"])
    for name in ("CMAKE_INSTALL_PREFIX", "LLVM_EXTERNAL_POLYGEIST_SOURCE_DIR", "POLYGEIST_PGO_DEFAULT_DATA_DIR"):
        assert name not in settings, f"{name} is a path the script sets under the scratch root"
    for name, key in (("CMAKE_C_COMPILER", "CLANG_C"), ("CMAKE_CXX_COMPILER", "CLANG_CXX")):
        if name in settings:
            assert settings[name].rsplit("/", 1)[-1] == values[key], f"the flags name {settings[name]} as {name}"


def test_pin_builds_the_tools_p5_calls() -> None:
    """cgeist, mlir-opt, mlir-translate, and clang, and nothing beyond polygeist-opt and clang's resource headers."""
    targets = pin()["TARGETS"].split()
    assert len(targets) == len(set(targets)), targets
    assert REQUIRED_TARGETS <= set(targets) <= ALLOWED_TARGETS, targets


def test_pin_names_each_host_tool_with_its_version() -> None:
    """CMake by path, Ninja, clang-17, and Python 3.10, each with the version the spike's probes read."""
    values = pin()
    assert (values["CMAKE"], values["CMAKE_EXPECT"]) in CMAKE_CHOICES
    assert (values["NINJA"], values["NINJA_EXPECT"]) == ("/usr/bin/ninja", "1.10.1")
    assert (values["CLANG_C"], values["CLANG_CXX"]) == ("clang-17", "clang++-17")
    assert "clang version 17.0.6" in values["CLANG_EXPECT"]
    assert (values["PYTHON"], values["PYTHON_EXPECT"]) == ("/usr/bin/python3.10", "Python 3.10.12")


def test_pin_names_the_built_tools_in_one_bin_directory() -> None:
    """Each tool lies in the prefix, in the in-tree build's one bin directory, where cgeist finds clang's headers."""
    values = pin()
    parents = set()
    for key, name in TOOLS.items():
        path = values[key]
        assert path and not path.startswith("/") and ".." not in path.split("/"), f"{key} lies inside the prefix"
        assert path.endswith(f"bin/{name}"), f"{key} is the build tree's bin/{name}"
        parents.add(path.rsplit("/", 1)[0])
    assert len(parents) == 1, f"one build tree, so <bin>/../lib/clang/18 is cgeist's resource directory: {parents}"


def test_pin_names_the_versions_the_tools_print() -> None:
    """EXPECT_VERSION is LLVM's banner line (cgeist, mlir-opt, mlir-translate); LLVM_CLANG_EXPECT is that clang's."""
    values = pin()
    assert LLVM_VERSION_LINE in values["EXPECT_VERSION"]
    match = CLANG_VERSION.fullmatch(values["LLVM_CLANG_EXPECT"])
    assert match, f"clang prints 'clang version 18.0.0', no git suffix: {values['LLVM_CLANG_EXPECT']!r}"
    if match.group(1):
        assert (match.group(1), match.group(2)) == (LLVM_URL, LLVM_COMMIT), "the origin URL and HEAD it was built from"


def test_pin_plans_space_for_both_stacks() -> None:
    """INSTALL_KIB is the reserve tt-mlir.pin holds for this stack, and RESERVE_KIB is tt-mlir's stack (PROJECTED)."""
    values, tt = pin(), read_pin(TT_MLIR_PIN_NAME)
    install, reserve = values["INSTALL_KIB"], values["RESERVE_KIB"]
    assert install.isdigit() and reserve.isdigit()
    assert int(install) >= kib_of(POLYGEIST_STACK_HUNDREDTHS), "19.15 GiB, the Polygeist stack (configuration A)"
    assert int(reserve) >= kib_of(TT_MLIR_STACK_HUNDREDTHS), "23.30 GiB, the tt-mlir stack (configuration A)"
    # Each pin plans the other stack with the figure the other pin plans for itself, so the two plans agree.
    assert int(install) == int(tt["RESERVE_KIB"]), "the space tt-mlir.pin holds for this stack is the space it plans"
    assert int(reserve) == int(tt["INSTALL_KIB"]), "the reserve held for tt-mlir is the space tt-mlir.pin plans"
    both = int(install) + int(reserve)
    assert both >= kib_of(BOTH_STACKS_HUNDREDTHS), "42.45 GiB, both stacks (configuration A)"
    assert both < STOP_KIB


def test_pin_values_match_the_bible_mlir_stacks() -> None:
    """The two commits are the ones the bible's Toolchain Pins section records."""
    pins = section(BIBLE.read_text(encoding="utf-8"), "### Toolchain Pins")
    values = pin()
    for key in ("COMMIT", "LLVM_COMMIT"):
        assert values[key] in pins, key


@needs_bash
def test_bash_reads_the_pin_as_read_pin_does(tmp_path: Path) -> None:
    """The shell that sources the pin sees the values read_pin gives (no unquoted ; or # in a value)."""
    values = pin()
    shutil.copy(TOOLCHAINS / "polygeist.pin", tmp_path / "polygeist.pin")
    body = "set -euo pipefail\nsource ./polygeist.pin\n" + "".join(f'printf "%s\\n" "${key}"\n' for key in values)
    result = run_probe(body, tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == list(values.values())


# Trial.toolchain_pins with two LLVMs (point 4 of the brief).


def test_toolchain_pins_tell_the_two_llvms_apart() -> None:
    """No schema change: llvm names the CPU stack's LLVM, polygeist and tt_mlir their pins, each value distinct.

    A pin reaches its field by its name with '-' made '_' (lassi.core.runner,
    _pin_versions), and records the pin's VERSION. The llvm field records
    polygeist.pin's LLVM_VERSION, which names 26eb4285 and never tt-mlir's
    4efe170d; tt-mlir.pin carries no such key, so tt-mlir's LLVM is named
    only through the tt_mlir pin, whose pin file records it as LLVM_COMMIT.
    """
    values, tt = pin(), read_pin(TT_MLIR_PIN_NAME)
    assert {"llvm", "polygeist", "tt_mlir"} <= set(TOOLCHAIN_PIN_NAMES)
    assert not [name for name in TOOLCHAIN_PIN_NAMES if "llvm" in name and name != "llvm"], "the field does not split"
    assert {PIN_NAME.replace("-", "_"), TT_MLIR_PIN_NAME.replace("-", "_")} == {"polygeist", "tt_mlir"}
    llvm = values["LLVM_VERSION"]
    assert LLVM_COMMIT[:8] in llvm and TT_LLVM_COMMIT[:8] not in llvm, llvm
    assert (values["LLVM_COMMIT"], tt["LLVM_COMMIT"]) == (LLVM_COMMIT, TT_LLVM_COMMIT)
    assert "LLVM_VERSION" not in tt, "tt-mlir's LLVM is named by the tt_mlir pin alone"
    record = ToolchainPins(llvm=llvm, polygeist=values["VERSION"], tt_mlir=tt["VERSION"])
    assert len({record.llvm, record.polygeist, record.tt_mlir}) == 3


# The tracked list and the files.


def test_tracked_fetched_sources_list_is_recorded_or_placeholder() -> None:
    """The configure's fetches are a tracked list: PLACEHOLDER until the first install, then its record."""
    path = TOOLCHAINS / TRACKED_FETCHED_NAME
    assert path.read_bytes().startswith(b"#"), "a comment header says where the list comes from"
    entries = entry_lines(path)
    if entries in (["PLACEHOLDER"], ["none"]):
        return
    assert entries, "an empty list cannot be told from a missing one; write none"
    for line in entries:
        assert FETCHED_ENTRY.fullmatch(line), f"{TRACKED_FETCHED_NAME}: not an entry: {line!r}"


@pytest.mark.parametrize("name", ["polygeist.pin", SCRIPT_NAME, TRACKED_FETCHED_NAME])
def test_files_are_plain_ascii_with_lf(name: str) -> None:
    """Plain ASCII and LF line endings, as every tracked toolchain file."""
    data = (TOOLCHAINS / name).read_bytes()
    assert data.isascii(), name
    assert b"\r" not in data, name


def test_no_file_matches_a_gate_device_pattern() -> None:
    """The gate refuses command text naming a disabled device class; these files stay clear of every pattern."""
    devices = json.loads(GATE_CONFIG.read_text(encoding="utf-8"))["devices"]
    patterns = [pattern for device in devices.values() for pattern in device["patterns"]]
    assert patterns
    for name in ("polygeist.pin", SCRIPT_NAME, TRACKED_FETCHED_NAME):
        text = (TOOLCHAINS / name).read_text(encoding="utf-8")
        for pattern in patterns:
            assert not re.search(pattern, text), f"{name} matches the gate pattern {pattern}"


@needs_bash
def test_script_parses_as_bash() -> None:
    """bash -n accepts the script."""
    assert (TOOLCHAINS / SCRIPT_NAME).is_file()
    assert BASH is not None
    # A relative POSIX path, since bash on Windows would read backslashes in a native path as escapes.
    result = subprocess.run([BASH, "-n", f"toolchains/{SCRIPT_NAME}"], capture_output=True, text=True, cwd=REPO)
    assert result.returncode == 0, result.stderr


# The script, read as text: where it may write and what it never runs.


# The core limit (Agent Rule 7). alpha01's core handler pipes a core to systemd-coredump, which ignores a limit of 0
# (`ulimit -c 0`) and stores the core on the root filesystem; the kernel aborts a piped dump only at exactly 1 byte
# (plans/spikes/p0-sandbox-hardening.md, R6, rx 20260923-133451-exec-7c3d). bash's `ulimit -c` counts 1024-byte
# blocks, so prlimit sets it and getrlimit reads it back (plans/LESSONS.md; toolchains/tt-mlir.sh).
CORE_LIMIT = "prlimit --pid $$ --core=1:1"
CORE_READ = "resource.getrlimit(resource.RLIMIT_CORE) != (1, 1)"
# A code line that runs a tool: a pinned host tool by its variable, or git, pip, CMake, Ninja, a download, a new
# directory or temporary file, readelf, or find.
TOOL_RUN = re.compile(
    r'"\$(?:CMAKE|NINJA|PYTHON|CLANG_C|CLANG_CXX)"'
    r"|\b(?:git|git_in|pip3?|cmake|ninja|curl|wget|mkdir|mktemp|readelf|find)\b"
)


def core_block(text: str) -> str:
    """Return the script's core-limit block, from its `if ! prlimit` line through its `fi` line."""
    start = text.index(f"\nif ! {CORE_LIMIT}") + 1
    return text[start : text.index("\nfi\n", start) + len("\nfi\n")]


# The root check's case line: the resolved root and a slash, matched against the allowed root, quoted so that a glob
# character in it is matched as itself, and anything under it (toolchains/tt-mlir.sh's).
ROOT_CASE = re.compile(r'case "\$\(realpath -m "\$root"\)/" in\n +"\$\{?[A-Za-z_][A-Za-z0-9_]*\}?"/\*\) ;;')


def test_script_is_strict_sources_its_pin_and_needs_the_gate_environment() -> None:
    """tt-mlir.sh's preamble: strict bash, the gate's variables and the host values, and the three roots allowed."""
    text = script_text()
    assert text.startswith("#!/usr/bin/env bash\n")
    assert "set -euo pipefail" in text
    assert CORE_LIMIT in text and CORE_READ in text
    # The allowed root and the stop line come from the host profile, or on alpha01 the command line, with no default.
    for name in ("LASSI_TOOLCHAINS", "LASSI_SCRATCH", "LASSI_JOBS", "LASSI_ALLOWED_ROOT", "LASSI_STOP_GIB", "HOME"):
        assert f'"${{{name}:?' in text, name
    loop = [line for line in logic_lines(text) if line.startswith("for root in ")]
    assert len(loop) == 1, loop
    for name in ("$LASSI_TOOLCHAINS", "$LASSI_SCRATCH", "$HOME"):
        assert f'"{name}"' in loop[0], f"{name} must lie in the allowed root (Agent Rule 7)"
    assert ROOT_CASE.search(text), "the root check's case pattern is the quoted allowed root, then /*"
    assert 'export TMPDIR="$LASSI_SCRATCH/tmp"' in text
    assert 'prefix="$LASSI_TOOLCHAINS/$PREFIX_NAME"' in text


# The allowed root and the stop line (task P5.15), as tests/toolchains/test_tt_mlir_pins.py checks tt-mlir.sh's.


ALPHA01_ROOT = "/mnt/nvme10/joseph_ufl"
# What alpha01's rx command line gives the script (Host Facts: the 145G stop line under the 150G cap), with the
# gate's layout of the roots (tools/server/gate.py, build_env), and HOME at the scratch root, as there.
ALPHA01 = {
    "LASSI_ALLOWED_ROOT": ALPHA01_ROOT,
    "LASSI_STOP_GIB": "145",
    "LASSI_TOOLCHAINS": f"{ALPHA01_ROOT}/toolchains",
    "LASSI_SCRATCH": ALPHA01_ROOT,
    "LASSI_JOBS": "4",
    "HOME": ALPHA01_ROOT,
}
# Another host's values, as a host profile gives them, under the probe's directory ("$PWD" in a value names it);
# SYNTHETIC.
LOCAL = {
    "LASSI_ALLOWED_ROOT": "$PWD/scratch",
    "LASSI_STOP_GIB": "20",
    "LASSI_TOOLCHAINS": "$PWD/scratch/toolchains",
    "LASSI_SCRATCH": "$PWD/scratch",
    "LASSI_JOBS": "4",
    "HOME": "$PWD/scratch",
}
# The values the probes unset before they set their own; HOME is always given.
SCRIPT_VARIABLES = ("LASSI_ALLOWED_ROOT", "LASSI_STOP_GIB", "LASSI_TOOLCHAINS", "LASSI_SCRATCH", "LASSI_JOBS")
# The tools the whole-script probes stand in for: prlimit, the first the script runs once its checks pass, and every
# tool that would write or fetch, so a script that ran one before its core limit would show it and write nothing.
STAND_INS = ("prlimit", "mkdir", "mktemp", "git", "curl", "cp", "mv", "ln", "rm")


def run_script(cwd: Path, values: dict[str, str]) -> tuple[subprocess.CompletedProcess[str], list[str], str]:
    """Run toolchains/polygeist.sh from `cwd` with only `values` of SCRIPT_VARIABLES (and HOME) set, under stand-ins.

    Each value is a double-quoted bash word, so "$PWD" in it names `cwd`. Each stand-in is an exported bash
    function that logs its call to calls.log and fails. Returns the result, the logged calls, and `cwd` as bash
    spells it.
    """
    lines = ["set -euo pipefail", 'printf "%s" "$PWD" >pwd.txt', ": >calls.log", 'export CALLS_LOG="$PWD/calls.log"']
    for tool in STAND_INS:
        lines += [f'{tool}() {{ echo "{tool} $*" >>"$CALLS_LOG"; return 97; }}', f"export -f {tool}"]
    lines.append("unset " + " ".join(SCRIPT_VARIABLES))
    lines += [f'export {key}="{value}"' for key, value in values.items()]
    lines.append(f"exec \"$BASH\" '{(TOOLCHAINS / SCRIPT_NAME).as_posix()}'")
    result = run_probe("\n".join(lines) + "\n", cwd)
    calls = (cwd / "calls.log").read_text(encoding="ascii").splitlines()
    return result, calls, (cwd / "pwd.txt").read_text(encoding="ascii")


def report_of(result: subprocess.CompletedProcess[str], calls: list[str]) -> str:
    """Return a whole-script probe's exit status, output, and logged calls, for an assertion message."""
    return f"exit {result.returncode}; stdout {result.stdout!r}; stderr {result.stderr!r}; calls {calls}"


def accepted(result: subprocess.CompletedProcess[str], calls: list[str]) -> None:
    """Assert the script passed its checks: nothing was refused, and its first step after them set the core limit."""
    report = report_of(result, calls)
    assert "refusing" not in result.stderr, report
    assert calls and calls[0].startswith("prlimit ") and "--core=1:1" in calls[0], (
        f"the first step past the checks must set the 1-byte core limit: {report}"
    )
    assert result.stdout == "", report


def refused(result: subprocess.CompletedProcess[str], calls: list[str], status: int | None = 2) -> str:
    """Assert the script stopped at its checks with `status` (None: any failure), ran nothing, printed nothing."""
    report = report_of(result, calls)
    assert result.returncode != 0 if status is None else result.returncode == status, report
    assert calls == [], f"a tool ran before the refusal: {report}"
    assert result.stdout == "", report
    return result.stderr


def says_where_the_values_come_from(stderr: str) -> None:
    """Assert a refusal says the values come from the host profile or, on alpha01, the command line."""
    for word in ("host-profile.sh", "alpha01"):
        assert word in stderr, f"the message names {word}: {stderr}"


@needs_bash
@pytest.mark.parametrize(
    ("changed", "kept"),
    [
        ({}, True),
        ({"LASSI_TOOLCHAINS": ALPHA01_ROOT, "LASSI_SCRATCH": f"{ALPHA01_ROOT}/"}, True),
        ({"LASSI_TOOLCHAINS": f"{ALPHA01_ROOT}/x/../toolchains"}, True),
        ({"HOME": f"{ALPHA01_ROOT}/home"}, True),
        ({"LASSI_TOOLCHAINS": "/mnt/nvme10/other/toolchains"}, False),
        ({"LASSI_TOOLCHAINS": f"{ALPHA01_ROOT}/../other"}, False),
        ({"LASSI_TOOLCHAINS": f"{ALPHA01_ROOT}2/toolchains"}, False),
        ({"LASSI_SCRATCH": "/mnt/nvme10/other"}, False),
        ({"HOME": "/home/someone"}, False),
    ],
    ids=["gate-layout", "the-root-itself", "resolves-inside", "home-inside", "outside", "resolves-outside", "sibling",
         "scratch-outside", "home-outside"],
)
def test_script_keeps_and_refuses_with_alpha01s_values(tmp_path: Path, changed: dict[str, str], kept: bool) -> None:
    """With alpha01's root and 145 on its command line, roots and HOME under the root are kept, no others."""
    result, calls, _ = run_script(tmp_path, {**ALPHA01, **changed})
    if kept:
        accepted(result, calls)
        return
    stderr = refused(result, calls)
    assert f"refusing to work outside {ALPHA01_ROOT}" in stderr, stderr
    says_where_the_values_come_from(stderr)


@needs_bash
@pytest.mark.parametrize(
    ("changed", "kept"),
    [
        ({}, True),
        ({"LASSI_TOOLCHAINS": f"{ALPHA01_ROOT}/toolchains"}, False),
        ({"LASSI_TOOLCHAINS": "$PWD/scratch2/toolchains"}, False),
        ({"HOME": ALPHA01_ROOT}, False),
    ],
    ids=["inside", "alpha01s-root", "sibling", "home-on-alpha01s-root"],
)
def test_script_takes_another_allowed_root_and_stop_line(tmp_path: Path, changed: dict[str, str], kept: bool) -> None:
    """A host profile's root and stop line work as alpha01's do; alpha01's root is then outside the allowed one."""
    result, calls, pwd = run_script(tmp_path, {**LOCAL, **changed})
    if kept:
        accepted(result, calls)
        return
    stderr = refused(result, calls)
    assert f"refusing to work outside {pwd}/scratch" in stderr, stderr
    says_where_the_values_come_from(stderr)


@needs_bash
@pytest.mark.parametrize(("under", "kept"), [("scr*tch", True), ("scrXtch", False)], ids=["the-root", "glob-match"])
def test_script_matches_the_allowed_root_literally(tmp_path: Path, under: str, kept: bool) -> None:
    """A glob character in the allowed root is matched as itself, so a root it would match as a pattern is refused."""
    values = {**LOCAL, "LASSI_ALLOWED_ROOT": "$PWD/scr*tch", "LASSI_TOOLCHAINS": f"$PWD/{under}/toolchains",
              "LASSI_SCRATCH": f"$PWD/{under}", "HOME": f"$PWD/{under}"}
    result, calls, pwd = run_script(tmp_path, values)
    if kept:
        accepted(result, calls)
        return
    stderr = refused(result, calls)
    assert f"refusing to work outside {pwd}/scr*tch" in stderr, stderr


@needs_bash
@pytest.mark.parametrize("variable", ["LASSI_ALLOWED_ROOT", "LASSI_STOP_GIB"])
@pytest.mark.parametrize("empty", [False, True], ids=["unset", "empty"])
def test_script_refuses_without_the_allowed_root_or_the_stop_line(tmp_path: Path, variable: str, empty: bool) -> None:
    """Neither value has a default (AGENTS.md, rules for [OPEN] choices): unset or empty, the script stops first."""
    values = {key: value for key, value in ALPHA01.items() if key != variable}
    if empty:
        values[variable] = ""
    result, calls, _ = run_script(tmp_path, values)
    stderr = refused(result, calls, status=None)
    assert variable in stderr, stderr
    says_where_the_values_come_from(stderr)


@needs_bash
@pytest.mark.parametrize("allowed", ["/", "scratch", "./scratch"], ids=["slash", "relative", "dot-relative"])
def test_script_refuses_an_allowed_root_that_is_relative_or_slash(tmp_path: Path, allowed: str) -> None:
    """"/" would allow the whole file system, and a relative root would move with the working directory."""
    # The roots given lie under both readings of the allowed root ($PWD/scratch).
    result, calls, _ = run_script(tmp_path, {**LOCAL, "LASSI_ALLOWED_ROOT": allowed})
    stderr = refused(result, calls)
    assert "LASSI_ALLOWED_ROOT" in stderr, stderr


@needs_bash
@pytest.mark.parametrize("value", ["0", "abc", "-1", "1.5", "145G", "0145"])
def test_script_refuses_a_stop_line_that_is_not_a_positive_integer(tmp_path: Path, value: str) -> None:
    """LASSI_STOP_GIB is checked with the other values, before the core limit or any tool, and the message prints it."""
    result, calls, _ = run_script(tmp_path, {**ALPHA01, "LASSI_STOP_GIB": value})
    stderr = refused(result, calls)
    assert "LASSI_STOP_GIB" in stderr and f"'{value}'" in stderr, stderr


@needs_bash
@pytest.mark.parametrize("value", ["0", "x", ""], ids=["zero", "word", "empty"])
def test_script_refuses_a_job_count_that_is_not_a_positive_integer(tmp_path: Path, value: str) -> None:
    """LASSI_JOBS, the -j of every build, is a positive integer, checked before the core limit or any tool."""
    result, calls, _ = run_script(tmp_path, {**ALPHA01, "LASSI_JOBS": value})
    stderr = refused(result, calls, status=None)
    assert "LASSI_JOBS" in stderr, stderr


def test_script_names_no_allowed_root_of_its_own() -> None:
    """alpha01's root is a value its command line gives, not a literal of the script; a message may still name it."""
    lines = [line for line in code_lines(script_text()) if ALPHA01_ROOT in line]
    messages = re.compile(r'(\*\) )?echo |"|: "\$\{[A-Z_]+:\?')
    assert all(messages.match(line) for line in lines), lines


def header(text: str) -> str:
    """Return a script's header comment: the comment lines after the shebang, without their #, joined by spaces."""
    found = []
    for line in text.splitlines()[1:]:
        if not line.startswith("#"):
            break
        found.append(line.lstrip("#").strip())
    return " ".join(found)


def test_header_says_how_the_script_runs_on_each_host() -> None:
    """On the workstation it runs under the host profile through the long-job runner; on alpha01 through rx."""
    head = header(script_text())
    needed = ("tools/local/host-profile.sh", "tools/local/longjob.py", "tools/rx.py", "alpha01", "LASSI_ALLOWED_ROOT",
              "LASSI_STOP_GIB")
    assert not [word for word in needed if word not in head], head


def test_script_holds_the_core_limit_at_one_byte_first() -> None:
    """prlimit sets RLIMIT_CORE to 1:1 and getrlimit reads it back, after the root checks, before any fetch or build."""
    text = script_text()
    assert not [line for line in logic_lines(text) if re.match(r"ulimit\b", line)], "ulimit -c counts 1024-byte blocks"
    assert text.count(CORE_LIMIT) == 1
    block = core_block(text)
    start = text.index(block)
    read = re.search(r"\"\$PYTHON\" -I -c '([^']+)'", block)
    assert read and CORE_READ in read.group(1), "the pinned Python, isolated, reads the limit back"
    compile(read.group(1), "core-read", "exec")
    assert block.rstrip().endswith("exit 2\nfi"), "a limit not set, or not read as (1, 1), stops the script"
    roots = text.index('for root in "$LASSI_TOOLCHAINS" "$LASSI_SCRATCH" "$HOME"; do')
    assert text.index("\ndone\n", roots) < start, "the root checks refuse first"
    assert not [line for line in logic_lines(text[:start]) if TOOL_RUN.search(line)], "no tool runs before it"
    main_start = text.index("\ntrap cleanup EXIT\n")
    first_function = min(match.start() for match in FUNCTION.finditer(text))
    assert start < first_function < main_start, "every function and main step, so every tool run, comes after it"
    assert start < text.index('mkdir -p "$TMPDIR"')
    assert start < main_start + first_download(text)


@needs_bash
@pytest.mark.parametrize(
    ("set_status", "read_status", "holds"),
    [(0, 0, True), (1, 0, False), (0, 1, False)],
    ids=["held", "prlimit-fails", "reads-otherwise"],
)
def test_core_limit_block_refuses_unless_set_and_read_back(
    tmp_path: Path, set_status: int, read_status: int, holds: bool
) -> None:
    """The block exits 2 with one polygeist line when prlimit fails or the limit does not read (1, 1)."""
    stubs = f"prlimit() {{ return {set_status}; }}\nread_back() {{ return {read_status}; }}\nPYTHON=read_back\n"
    result = run_probe("set -euo pipefail\n" + stubs + core_block(script_text()) + "echo held\n", tmp_path)
    if holds:
        assert (result.returncode, result.stdout) == (0, "held\n"), result.stdout + result.stderr
        return
    assert result.returncode == 2 and result.stdout == "", result.stdout + result.stderr
    lines = result.stderr.splitlines()
    assert len(lines) == 1 and lines[0].startswith("polygeist:") and "1 byte" in lines[0], result.stderr


def test_script_never_runs_a_submodule_fetch_sudo_or_pip() -> None:
    """llvm-project comes by commit at depth 1, never `git submodule` (its whole history); no sudo and no pip."""
    lines = logic_lines(script_text())
    for word in (r"\bsubmodule\b", r"--recursive\b", r"--recurse-submodules\b", r"\bsudo\b", r"\bpip3?\b"):
        assert not [line for line in lines if re.search(word, line)], word
    sourced = [line for line in lines if re.search(r'(^|[\s;(&|])(source|\.)\s+"?\$', line)]
    assert 'source "$here/polygeist.pin"' in sourced, "the script sources its pin"
    for line in sourced:
        assert "/polygeist.pin" in line or "/tt-mlir.pin" in line, f"only the two pins are sourced: {line}"


def test_script_fetches_both_sources_by_commit_at_depth_one() -> None:
    """Polygeist at COMMIT from URL and llvm-project at LLVM_COMMIT from LLVM_URL, each with --depth 1."""
    lines = code_lines(script_text())
    for url, commit in (('"$URL"', '"$COMMIT"'), ('"$LLVM_URL"', '"$LLVM_COMMIT"')):
        found = [line for line in lines if url in line and commit in line and "--depth 1" in line]
        assert found, f"no fetch of {commit} from {url} at depth 1"


def test_script_writes_no_system_path() -> None:
    """Only the root-filesystem check names /tmp, /var/tmp, or /opt; no command uses a /usr/, /etc/, or /opt/ path."""
    shebang, _, rest = script_text().partition("\n")
    assert shebang == "#!/usr/bin/env bash"
    commands = "\n".join(logic_lines(rest))
    for system_path in ("/usr/", "/opt/", "/etc/"):
        assert system_path not in commands, system_path
    naming = [line for line in logic_lines(rest) if re.search(r'(^|[\s"=(])/((var/)?tmp|opt)\b', line)]
    assert naming == ['check_root_fs "$marker" /tmp /var/tmp /opt']


def test_script_writes_nothing_under_home() -> None:
    """HOME is checked to lie in the scratch root and is not otherwise named, so nothing is written there."""
    lines = logic_lines(script_text())
    named = [line for line in lines if "$HOME" in line or "${HOME" in line]
    assert len(named) == 2, named
    assert [line for line in named if line.startswith("for root in ")]


def test_root_fs_check_is_the_one_tt_metal_sh_uses() -> None:
    """The P4.2 method: the same find over the named directories, read the same way (messages aside)."""
    ours = logic_lines(bash_function(script_text(), "check_root_fs"))
    theirs = logic_lines(bash_function((TOOLCHAINS / "tt-metal.sh").read_text(encoding="utf-8"), "check_root_fs"))
    assert ours == theirs


def test_script_clears_the_build_environment() -> None:
    """No compiler, search-path, Python, or CMake variable of the caller reaches the build."""
    text = script_text()
    unset = [line for line in logic_lines(text) if line.startswith("unset CC CXX ")]
    assert len(unset) == 1, unset
    names = set(unset[0].split()) - {"unset"}
    required = {"CC", "CXX", "CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS", "CPATH", "C_INCLUDE_PATH",
                "CPLUS_INCLUDE_PATH", "LIBRARY_PATH", "LD_LIBRARY_PATH", "CMAKE_PREFIX_PATH", "PKG_CONFIG_PATH",
                "PYTHONPATH", "PYTHONHOME"}
    assert required <= names, required - names
    loop = text.index("$(compgen -e)")
    assert "CMAKE_*" in text[loop : text.index("esac", loop)]


def test_script_names_cmake_by_path_and_builds_with_the_gate_job_count() -> None:
    """CMake runs as "$CMAKE", Ninja is named to it, and every build runs at LASSI_JOBS."""
    lines = logic_lines(script_text())
    assert not [line for line in lines if re.search(r"(^|[\s;&|(])cmake\s", line)], "no cmake from PATH"
    assert [line for line in lines if '"$CMAKE"' in line]
    assert [line for line in code_lines(script_text()) if "-DCMAKE_MAKE_PROGRAM=" in line and "$NINJA" in line]
    builds = [line for line in lines if "--build" in line]
    assert builds and all('--parallel "$LASSI_JOBS"' in line for line in builds), builds


def test_script_names_the_pinned_compilers_to_the_configure() -> None:
    """clang-17 and clang++-17 by name, unless the pin's flags already name them."""
    settings = cmake_settings(pin()["LLVM_CMAKE_FLAGS"])
    if {"CMAKE_C_COMPILER", "CMAKE_CXX_COMPILER"} <= set(settings):
        return
    lines = code_lines(script_text())
    assert [line for line in lines if re.search(r'-DCMAKE_C_COMPILER="?\$\{?CLANG_C\b', line)]
    assert [line for line in lines if re.search(r'-DCMAKE_CXX_COMPILER="?\$\{?CLANG_CXX\b', line)]


def test_script_builds_the_pinned_targets_and_installs_nothing() -> None:
    """The build names the pin's TARGETS; nothing builds all or install, and no install prefix is set."""
    text = script_text()
    direct = [line for line in logic_lines(main_section(text)) if "--build" in line and "$TARGETS" in line]
    assert builders(text) or direct, "a build names $TARGETS"
    lines = logic_lines(text)
    installs = re.compile(r"--target\s+(all|install)\b|\bninja\s+install\b|--install\b")
    assert not [line for line in lines if installs.search(line)]
    assert not [line for line in code_lines(text) if "CMAKE_INSTALL_PREFIX" in line], "LLVM is never installed"


def test_script_checks_the_configured_cache() -> None:
    """After the configure, check_cache reads the pin's flags and the script's own settings back from the cache."""
    checks = [line for line in logic_lines(script_text()) if calls(line, {"check_cache"})]
    assert [line for line in checks if '"$LLVM_CMAKE_FLAGS"' in line], "no check_cache of the pin's flags"
    assert [line for line in checks if "llvm_settings" in line], "no check_cache of the script's settings"


def test_check_host_checks_every_pinned_tool() -> None:
    """check_host refuses a host tool whose version is not the pinned one."""
    body = "\n".join(body_lines(script_text(), "check_host"))
    for key in ("CMAKE_EXPECT", "NINJA_EXPECT", "CLANG_EXPECT", "PYTHON_EXPECT"):
        assert f'"${key}"' in body, key


def test_script_verifies_both_sources_at_their_commits() -> None:
    """verify_clean checks Polygeist at COMMIT and llvm-project at LLVM_COMMIT, with no tracked file changed."""
    checks = [line for line in logic_lines(script_text()) if calls(line, {"verify_clean"})]
    assert [line for line in checks if '"$COMMIT"' in line], checks
    assert [line for line in checks if '"$LLVM_COMMIT"' in line], checks


# The script's steps, in order.


def test_kept_install_is_checked_first_and_changes_nothing() -> None:
    """A rerun at the pin keeps a checked install and exits before any marker, check, or download."""
    text = script_text()
    main = main_section(text)
    kept = main.index("already holds the pinned")
    condition = main[main.rindex("\nif ", 0, kept) : kept]
    assert '"$prefix/$record_name"' in condition and '"$(pinned_record)"' in condition
    block = main[kept : main.index("\nfi\n", kept)]
    assert "exit 0" in block
    assert "verify_install" in main[main.rindex("\nif ", 0, kept) : kept + len(block)]
    assert kept < main.index("mktemp")
    for step in ("check_prefix", "tt_mlir_installed", "check_host", "check_space"):
        assert kept < call_position(main, {step}), step
    assert kept < first_download(text)
    verify = "\n".join(body_lines(text, "verify_install"))
    assert "check_fetched_sources" in verify and '"$EXPECT_VERSION"' in verify


def test_every_refusal_comes_before_any_download() -> None:
    """The prefix, host, and space checks, with the tt-mlir read, run before the first fetch; so does the marker."""
    text = script_text()
    main, download = main_section(text), first_download(text)
    for step in ("check_prefix", "tt_mlir_installed", "check_host", "check_space"):
        assert call_position(main, {step}) < download, f"{step} runs after a download"
    assert main.index('pinned_record >"$prefix/$unfinished_name"') < download


def test_space_check_plans_both_stacks_together() -> None:
    """INSTALL_KIB alone once tt-mlir's install is complete; INSTALL_KIB plus RESERVE_KIB otherwise.

    The plan is the main steps' one if-block on tt_mlir_installed: its then
    branch is the one check_space of INSTALL_KIB, its else branch the one
    check_space of the sum, and no other line calls check_space.
    """
    main = main_section(script_text())
    lines = [line for _, line in positioned(main)]
    spaces = [line for line in lines if calls(line, {"check_space"})]
    assert spaces == ['check_space "$INSTALL_KIB"', 'check_space "$((INSTALL_KIB + RESERVE_KIB))"'], spaces
    start = lines.index("if tt_mlir_installed; then")
    assert lines[start : start + 5] == [
        "if tt_mlir_installed; then",
        'check_space "$INSTALL_KIB"',
        "else",
        'check_space "$((INSTALL_KIB + RESERVE_KIB))"',
        "fi",
    ], lines[start : start + 5]
    assert call_position(main, {"check_prefix"}) < call_position(main, {"tt_mlir_installed"})


@needs_bash
@pytest.mark.parametrize(("installed", "planned"), [(True, "INSTALL_KIB"), (False, "BOTH")], ids=["after", "before"])
def test_space_plan_block_runs_the_check_with_the_right_figure(tmp_path: Path, installed: bool, planned: str) -> None:
    """The plan block, run with stand-ins, hands check_space INSTALL_KIB or INSTALL_KIB + RESERVE_KIB."""
    main = main_section(script_text())
    start = main.index("\nif tt_mlir_installed; then\n") + 1
    block = main[start : main.index("\nfi\n", start) + len("\nfi\n")]
    stubs = (f"tt_mlir_installed() {{ return {0 if installed else 1}; }}\n"
             'check_space() { printf "%s\\n" "$1" >planned.txt; }\n')
    result = run_with_script(stubs + block, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    values = pin()
    expected = int(values["INSTALL_KIB"]) + (0 if planned == "INSTALL_KIB" else int(values["RESERVE_KIB"]))
    assert (tmp_path / "planned.txt").read_text(encoding="ascii") == f"{expected}\n"


def test_gitlink_is_checked_before_the_configure() -> None:
    """Polygeist's tree records LLVM_COMMIT as its llvm-project gitlink before anything is configured."""
    text = script_text()
    main = main_section(text)
    checked = call_position(main, {"check_gitlink"}, after=first_download(text))
    assert checked < call_position(main, {"record_fetched_sources"}, after=checked)


def test_script_records_compares_builds_verifies_and_records_in_order() -> None:
    """Fetched sources are recorded and compared before the build; the tools are checked; the record comes last."""
    text = script_text()
    main = main_section(text)
    recorded = call_position(main, {"record_fetched_sources"}, after=first_download(text))
    compared = call_position(main, {"check_fetched_sources"}, after=recorded)
    built = call_position(main, builders(text), after=compared) if builders(text) else main.index("$TARGETS", compared)
    version = step_position(text, '"$EXPECT_VERSION"', built)
    clang = step_position(text, '"$LLVM_CLANG_EXPECT"', built)
    checked = main.index('check_root_fs "$marker" /tmp /var/tmp /opt', max(version, clang))
    assert main.index('mv -T "$prefix/$unfinished_name" "$prefix/$record_name"', checked) > checked
    assert TREE_FETCHED_NAME in text and TRACKED_FETCHED_NAME in text


# Behavior: single functions of the script, run in bash on local files.


SPACE_STUBS = """du() { if [ -n "$DU_KIB" ]; then printf '%s\\t%s\\n' "$DU_KIB" "${@: -1}"; fi; }
df() { printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\\n/dev/x 1 1 %s 1%% /mnt\\n' "$DF_KIB"; }
LASSI_SCRATCH=/mnt/nvme10/scratch
"""


def space_body(stop_gib: int, planned: str = "1000") -> str:
    """Return a check_space probe body with the du and df stand-ins and LASSI_STOP_GIB set to `stop_gib`."""
    return SPACE_STUBS + f'LASSI_STOP_GIB={stop_gib}\ncheck_space "{planned}"\n'


@needs_bash
@pytest.mark.parametrize(
    ("used", "free", "fits"),
    [(STOP_KIB - 1000, 10**9, True), (STOP_KIB - 999, 10**9, False), (1000, 999, False), ("", 10**9, False)],
    ids=["at-the-stop-line", "over-the-stop-line", "disk-full", "du-fails"],
)
def test_check_space_stops_at_the_plan_line(tmp_path: Path, used: int | str, free: int, fits: bool) -> None:
    """With alpha01's LASSI_STOP_GIB=145, du plus the planned KiB may reach 145 GiB, never pass it, and must fit."""
    result = run_functions(("check_space",), space_body(145), tmp_path, env={"DU_KIB": str(used), "DF_KIB": str(free)})
    assert (result.returncode == 0) == fits, result.stdout + result.stderr
    if not fits:
        assert "polygeist:" in result.stderr


@needs_bash
@pytest.mark.parametrize("stop_gib", [145, 20])
@pytest.mark.parametrize(("below", "fits"), [(1000, True), (999, False)], ids=["at-the-stop-line", "over-it"])
def test_check_space_takes_its_stop_line_from_lassi_stop_gib(
    tmp_path: Path, stop_gib: int, below: int, fits: bool
) -> None:
    """The stop line is LASSI_STOP_GIB GiB (20 is SYNTHETIC, so a literal 145 shows), and the messages print it."""
    stop_kib = stop_gib * GIB_KIB
    result = run_functions(("check_space",), space_body(stop_gib), tmp_path,
                           env={"DU_KIB": str(stop_kib - below), "DF_KIB": str(10**9)})
    assert (result.returncode == 0) == fits, result.stdout + result.stderr
    if fits:
        assert f"stop line {stop_kib} KiB" in result.stdout, result.stdout
        return
    assert f"{stop_gib} GiB" in result.stderr and f"{stop_kib} KiB" in result.stderr, "the message prints the value"
    assert stop_gib == 145 or "145 GiB" not in result.stderr, result.stderr


@needs_bash
@pytest.mark.parametrize("planned", ["", "12x", "-5"], ids=["empty", "word", "negative"])
def test_check_space_refuses_a_plan_that_is_not_a_count(tmp_path: Path, planned: str) -> None:
    """A planned figure that is not a whole count of KiB is refused, never read as 0."""
    result = run_functions(("check_space",), space_body(145, planned), tmp_path,
                           env={"DU_KIB": "1000", "DF_KIB": str(10**9)})
    assert result.returncode != 0 and "polygeist:" in result.stderr, result.stdout + result.stderr


def comment_above(text: str, offset: int) -> str:
    """Return the comment lines right above the line that starts at `offset`, without their #, joined by spaces."""
    found = []
    for line in reversed(text[:offset].splitlines()):
        if not line.startswith("#"):
            break
        found.append(line.lstrip("#").strip())
    return " ".join(reversed(found))


def test_check_space_has_no_stop_line_of_its_own() -> None:
    """check_space reads LASSI_STOP_GIB, and its comment says so (145 on both hosts, Host Facts); no literal 145."""
    text = script_text()
    body = bash_function(text, "check_space")
    assert "LASSI_STOP_GIB" in body and "145" not in body, body
    comment = comment_above(text, text.index(body))
    assert "LASSI_STOP_GIB" in comment and "stop line of 145 GiB" not in comment, comment


@needs_bash
def test_pinned_record_holds_every_pinned_value(tmp_path: Path) -> None:
    """The record a kept install is compared with changes with any pinned commit, flag, target, tool, or version."""
    result = run_with_script("pinned_record >rec1.txt\npinned_record >rec2.txt\n", tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    record = (tmp_path / "rec1.txt").read_text(encoding="ascii")
    assert record == (tmp_path / "rec2.txt").read_text(encoding="ascii"), "the record is the same each time"
    assert record.splitlines()[0].split()[:2] == [PIN_NAME, POLYGEIST_COMMIT], "NAME and COMMIT on the first line"
    values = pin()
    keys = ("URL", "VERSION", "LLVM_COMMIT", "LLVM_URL", "LLVM_CMAKE_FLAGS", "TARGETS", "CMAKE", "CLANG_C",
            "CLANG_CXX", "PYTHON", *TOOLS, "EXPECT_VERSION", "LLVM_CLANG_EXPECT")
    for key in keys:
        assert values[key] in record, key
    assert not (tmp_path / "tc").exists(), "pinned_record writes nothing"


def settings_and_roots(tmp_path: Path) -> tuple[dict[str, str], list[str], str, str]:
    """Run pinned_record and llvm_settings; return the record's lines by first word, the settings, prefix, scratch."""
    body = ('pinned_record >rec.txt\nllvm_settings >llvm.txt\nprintf "%s\\n" "$prefix" "$LASSI_SCRATCH" >roots.txt\n')
    result = run_with_script(body, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    lines = (tmp_path / "rec.txt").read_text(encoding="ascii").splitlines()
    record = {line.split(" ", 1)[0]: line for line in lines}
    settings = (tmp_path / "llvm.txt").read_text(encoding="ascii").splitlines()
    prefix, scratch = (tmp_path / "roots.txt").read_text(encoding="ascii").splitlines()
    return record, settings, prefix, scratch


def setting_value(settings: list[str], name: str) -> str:
    """Return the value of the one -D<name>= setting in `settings`; fail when there is not exactly one."""
    found = [line.split("=", 1)[1] for line in settings if line.split("=", 1)[0].split(":", 1)[0] == f"-D{name}"]
    assert len(found) == 1, f"-D{name} is set {len(found)} times: {settings}"
    return found[0]


@needs_bash
def test_pinned_record_holds_the_settings_the_script_adds(tmp_path: Path) -> None:
    """llvm_settings' -D lines are one record line, as tt-mlir.sh records its own (llvm-local)."""
    record, settings, _, _ = settings_and_roots(tmp_path)
    assert settings and all(setting.startswith("-D") for setting in settings), settings
    # tr turns each setting's newline into one space (toolchains/tt-mlir.sh, pinned_record).
    assert record.get("llvm-local") == "llvm-local " + "".join(f"{setting} " for setting in settings)
    assert f"-DCMAKE_MAKE_PROGRAM={pin()['NINJA']}" in settings, "Ninja by path"
    assert "-DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF" in settings
    assert setting_value(settings, "Python3_EXECUTABLE") == pin()["PYTHON"], "the pinned Python, not one from PATH"


@needs_bash
def test_settings_place_polygeist_and_its_pgo_directory_in_the_scratch_root(tmp_path: Path) -> None:
    """Polygeist's source lies in the prefix, and the PGO directory (default /var/tmp/polygeist/pgo/) in scratch."""
    _, settings, prefix, scratch = settings_and_roots(tmp_path)
    source = setting_value(settings, "LLVM_EXTERNAL_POLYGEIST_SOURCE_DIR")
    assert source.startswith(prefix + "/"), f"README.md:92's source dir lies in the prefix: {source}"
    # CMakeLists.txt:44 caches the default; lib/polygeist/Passes/CMakeLists.txt:76-81 compiles it in, and the PGO
    # passes read it when POLYGEIST_PGO_DATA_DIR is unset (LowerAlternatives.cpp:50-56, joined with "/" at :64).
    pgo = setting_value(settings, "POLYGEIST_PGO_DEFAULT_DATA_DIR")
    assert pgo.startswith(scratch + "/") and "/var/tmp" not in pgo, pgo


@needs_bash
@pytest.mark.parametrize(
    ("case", "kept"),
    [("absent", True), ("resume", True), ("other-commit", False), ("other-llvm", False), ("unknown", False)],
)
def test_check_prefix_keeps_only_a_fresh_or_resumable_prefix(tmp_path: Path, case: str, kept: bool) -> None:
    """An unfinished tree at the pin resumes; any other install or content at the prefix is refused, never moved."""
    made = run_with_script("pinned_record >rec.txt\n", tmp_path)
    assert made.returncode == 0, made.stdout + made.stderr
    record = (tmp_path / "rec.txt").read_text(encoding="ascii")
    prefix = tmp_path / "tc" / pin()["PREFIX_NAME"]
    contents = {
        "resume": {UNFINISHED_NAME: record},
        "other-commit": {RECORD_NAME: record.replace(POLYGEIST_COMMIT, SYNTHETIC_COMMIT)},
        "other-llvm": {UNFINISHED_NAME: record.replace(LLVM_COMMIT, SYNTHETIC_COMMIT)},
        "unknown": {"stray.txt": "SYNTHETIC stray file\n"},
    }.get(case, {})
    for name, content in contents.items():
        prefix.mkdir(parents=True, exist_ok=True)
        (prefix / name).write_bytes(content.encode("ascii"))
    before = prefix_state(prefix)
    result = run_with_script("check_prefix\n", tmp_path)
    assert (result.returncode == 0) == kept, result.stdout + result.stderr
    assert prefix_state(prefix) == before, "check_prefix changes nothing at the prefix"
    if not kept:
        assert pin()["PREFIX_NAME"] in result.stderr and "never delete" in result.stderr


def tt_mlir_record(commit: str) -> str:
    """Return a SYNTHETIC tt-mlir install record whose first line names tt-mlir and `commit`, as tt-mlir.sh has it."""
    tt = read_pin(TT_MLIR_PIN_NAME)
    return f"tt-mlir {commit} {tt['URL']} {tt['VERSION']}\nSYNTHETIC rest of the record\n"


@needs_bash
@pytest.mark.parametrize(
    ("case", "passes"),
    [("installed", True), ("absent", False), ("unfinished", False), ("other-commit", False),
     ("record-and-unfinished", False)],
)
def test_tt_mlir_installed_reads_only_a_complete_install(tmp_path: Path, case: str, passes: bool) -> None:
    """Only tt-mlir's complete install record at tt-mlir's pinned commit counts as installed (point 3).

    The check reads the tt-mlir prefix only: it changes nothing there and no
    value of polygeist.pin (the probe prints PREFIX_NAME and COMMIT after it).
    It never refuses the install: either way it says which plan follows.
    """
    tt = read_pin(TT_MLIR_PIN_NAME)
    tt_prefix = tmp_path / "tc" / tt["PREFIX_NAME"]
    files = {
        "installed": {RECORD_NAME: tt_mlir_record(tt["COMMIT"])},
        "unfinished": {UNFINISHED_NAME: tt_mlir_record(tt["COMMIT"])},
        "other-commit": {RECORD_NAME: tt_mlir_record(SYNTHETIC_COMMIT)},
        "record-and-unfinished": {RECORD_NAME: tt_mlir_record(tt["COMMIT"]), UNFINISHED_NAME: "SYNTHETIC\n"},
    }.get(case, {})
    for name, content in files.items():
        tt_prefix.mkdir(parents=True, exist_ok=True)
        (tt_prefix / name).write_bytes(content.encode("ascii"))
    before = prefix_state(tt_prefix)
    body = ('if tt_mlir_installed; then echo "status 0"; else echo "status 1"; fi\n'
            'printf "%s %s\\n" "$PREFIX_NAME" "$COMMIT"\n')
    result = run_with_script(body, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert prefix_state(tt_prefix) == before, "the check changes nothing at tt-mlir's prefix"
    lines = result.stdout.splitlines()
    assert lines[-1] == f"{pin()['PREFIX_NAME']} {POLYGEIST_COMMIT}", "no value of polygeist.pin changes"
    assert lines[-2] == f"status {0 if passes else 1}", result.stdout + result.stderr
    message = lines[-3]
    assert message.startswith("polygeist:") and "tt-mlir" in message, result.stdout
    assert ("RESERVE_KIB" in message) != passes, "a tree that is not a complete install holds the reserve"


GITLINK_PROBE = """probe_repo() {
  local dir="$PWD/$1" sha="$2"
  git -c init.defaultBranch=main init -q "$dir"
  printf 'SYNTHETIC file for the gitlink probe\\n' >"$dir/README"
  git -C "$dir" add README
  if [ -n "$sha" ]; then
    git -C "$dir" update-index --add --cacheinfo "160000,$sha,$LLVM_SUBMODULE"
  fi
  git -C "$dir" -c commit.gpgsign=false commit -q -m "SYNTHETIC commit"
}
"""


@needs_bash
@pytest.mark.parametrize(
    ("gitlink", "passes"),
    [(LLVM_COMMIT, True), (SYNTHETIC_COMMIT, False), ("", False)],
    ids=["pinned", "other-commit", "no-gitlink"],
)
def test_check_gitlink_needs_llvm_commit_as_the_submodule(tmp_path: Path, gitlink: str, passes: bool) -> None:
    """check_gitlink <dir> passes only a Polygeist tree whose llvm-project gitlink is LLVM_COMMIT."""
    body = GITLINK_PROBE + f'probe_repo repo "{gitlink}"\ncheck_gitlink "$PWD/repo"\n'
    result = run_with_script(body, tmp_path)
    assert (result.returncode == 0) == passes, result.stdout + result.stderr
    if not passes:
        assert "polygeist:" in result.stderr and LLVM_COMMIT in result.stderr, result.stderr


ENTRY_A = "alpha/0001 git " + "1" * 40 + " changed=0"
ENTRY_B = "beta/0002 files-sha256 " + "2" * 64
TRACKED = f"# SYNTHETIC tracked list for the probe\n\n{ENTRY_A}\n{ENTRY_B}\n"


@needs_bash
@pytest.mark.parametrize(
    ("tree", "tracked", "passes"),
    [
        (f"{ENTRY_A}\n{ENTRY_B}  \n", TRACKED, True),
        ("none\n", "# SYNTHETIC tracked list\nnone\n", True),
        (f"{ENTRY_A.replace('changed=0', 'changed=1')}\n{ENTRY_B}\n", TRACKED, False),
        (f"{ENTRY_A}\n{ENTRY_B}\n" + "gamma/0003 git " + "3" * 40 + " changed=0\n", TRACKED, False),
        (f"{ENTRY_A}\n", TRACKED, False),
        (None, TRACKED, False),
        (f"{ENTRY_A}\n{ENTRY_B}\n", "# SYNTHETIC tracked list\nPLACEHOLDER\n", False),
    ],
    ids=["same", "none", "other-line", "added", "missing-line", "no-tree-list", "placeholder"],
)
def test_check_fetched_sources_compares_line_for_line(
    tmp_path: Path, tree: str | None, tracked: str, passes: bool
) -> None:
    """The tree's list equals the tracked one, comments, blanks, and trailing spaces aside; PLACEHOLDER is no record."""
    (tmp_path / "pins").mkdir()
    (tmp_path / "pins" / TRACKED_FETCHED_NAME).write_bytes(tracked.encode("ascii"))
    if tree is not None:
        (tmp_path / TREE_FETCHED_NAME).write_bytes(tree.encode("ascii"))
    body = f"check_fetched_sources {TREE_FETCHED_NAME} pins/{TRACKED_FETCHED_NAME}\n"
    result = run_functions(("check_fetched_sources",), body, tmp_path)
    assert (result.returncode == 0) == passes, result.stdout + result.stderr
    if not passes:
        assert "polygeist:" in result.stderr
        assert TRACKED_FETCHED_NAME in result.stderr or TREE_FETCHED_NAME in result.stderr, "the message names a list"
    if "PLACEHOLDER" in tracked:
        assert "PLACEHOLDER" in result.stderr


CACHE_FLAGS = "-G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS=clang;mlir -DPOLYGEIST_ENABLE_POLYMER=0"
CACHE_LINES = ["# SYNTHETIC CMake cache for the probe", "CMAKE_BUILD_TYPE:STRING=Release",
               "LLVM_ENABLE_PROJECTS:STRING=clang;mlir", "POLYGEIST_ENABLE_POLYMER:BOOL=0",
               "CMAKE_GENERATOR:INTERNAL=Ninja"]


@needs_bash
@pytest.mark.parametrize(
    ("lines", "key"),
    [
        (CACHE_LINES, None),
        ([line.replace("=Release", "=Debug") for line in CACHE_LINES], "CMAKE_BUILD_TYPE"),
        ([line.replace("POLYMER:BOOL=0", "POLYMER:BOOL=1") for line in CACHE_LINES], "POLYGEIST_ENABLE_POLYMER"),
        ([line for line in CACHE_LINES if not line.startswith("LLVM_")], "LLVM_ENABLE_PROJECTS"),
        (None, "CMakeCache.txt"),
    ],
    ids=["pinned", "other-value", "polymer-on", "missing-setting", "no-cache"],
)
def test_check_cache_refuses_a_setting_the_pin_does_not_record(
    tmp_path: Path, lines: list[str] | None, key: str | None
) -> None:
    """check_cache <cache> <flags> passes only a cache that holds every -D setting of the flags, value for value."""
    if lines is not None:
        (tmp_path / "b").mkdir()
        (tmp_path / "b" / "CMakeCache.txt").write_bytes(("\n".join(lines) + "\n").encode("ascii"))
    result = run_functions(("check_cache",), f"check_cache b/CMakeCache.txt '{CACHE_FLAGS}'\n", tmp_path)
    assert (result.returncode == 0) == (key is None), result.stdout + result.stderr
    if key is not None:
        assert key in result.stderr
