"""Tests for the tt-mlir pin file, its tracked lists, and its install script (task P5.4).

Bible: Toolchain Pins (the joint pin; MLIR stacks; pin files and install
rules), Target Rules, Host Facts (the 150G scratch cap and the 145G stop
line), Agent Rules 7 and 10; plans/p5-ir.md, P5.4; plans/spikes/p5-mlir-pins.md,
Results 2, 4, and 5 and "Consequences for the plan" (the P5.4 bullet is the
design).

The design: toolchains/tt-mlir.sh never runs tt-mlir's env/CMakeLists.txt or
env/activate. It builds flatbuffers at FLATBUFFERS_COMMIT into
TTMLIR_TOOLCHAIN_DIR (so flatc is found there) and llvm-project at
LLVM_COMMIT with tt-mlir's affine patch, checked by its git blob and sha256
before it is applied; then tt-mlir at COMMIT, cloned with its history so the
v0.0 tag its configure describes is local, with the runtime, OpModel,
Python bindings, tests, explorer, ttrt, and alchemist off and only
ttmlir-opt and ttmlir-translate built, so no second tt-metal is fetched or
built. A venv of the host's Python 3.10 holds sphinx (8.1.x or older),
sphinx-markdown-builder, and nanobind from a tracked requirements file with
a hash for every download, and its bin directory is first on PATH. The
script holds RLIMIT_CORE at exactly 1 byte once its root checks pass and
before anything is fetched, configured, or built, and it sets
TTMLIR_ENV_ACTIVATED, TTMLIR_TOOLCHAIN_DIR, and TTMLIR_VENV_DIR under the
install prefix $LASSI_TOOLCHAINS/<NAME>@<VERSION>, names CMake by path, and
builds with clang-17 and clang++-17.

The contract these tests fix, for the implementation to follow:

- toolchains/tt-mlir.pin keys: NAME, VERSION, PREFIX_NAME, COMMIT, URL;
  VERSION_TAG and VERSION_TAG_COMMIT; LLVM_COMMIT and LLVM_URL;
  FLATBUFFERS_COMMIT and FLATBUFFERS_URL; LLVM_PATCH (its path in tt-mlir's
  tree), LLVM_PATCH_BLOB, and LLVM_PATCH_SHA256; PYTHON_REQUIREMENTS;
  CMAKE_FLAGS (tt-mlir), LLVM_CMAKE_FLAGS, and FLATBUFFERS_CMAKE_FLAGS, each
  "-G Ninja" then -DKEY=VALUE words, no install prefix; TARGETS; the host
  tools CMAKE, NINJA, CLANG_C, CLANG_CXX, LINKER, and PYTHON, each with its
  *_EXPECT version text; TTMLIR_OPT and TTMLIR_TRANSLATE (paths relative to
  the prefix) and EXPECT_VERSION; TOOLCHAIN_DIR and VENV_DIR (relative to
  the prefix); INSTALL_KIB (this stack) and RESERVE_KIB (the Polygeist
  stack, held for P5.5), whose sum the space check plans for.
- toolchains/tt-mlir-requirements.txt: name==version lines, each with
  --hash=sha256 values.
- toolchains/tt-mlir-fetched-sources.txt: the tracked list of what the
  configure fetched, compared with the tree's lassi-fetched-sources.txt;
  its entries are the one line PLACEHOLDER until the first install records
  them, the one line none when the configure fetched nothing, or lines in
  tt-metal's two forms (toolchains/tt-metal-cpm-sources.txt).
- toolchains/tt-mlir.sh, in tt-metal.sh's style (`name() {` functions, the
  main steps after `trap cleanup EXIT`, the record lassi-install.txt and
  the marker lassi-install.unfinished): pinned_record, check_prefix,
  expect_version, check_host, check_space <KiB>, check_patch <file>,
  check_cache <CMakeCache.txt> <flags>, record_fetched_sources,
  check_fetched_sources <tree list> <tracked list>, verify_install, and
  check_root_fs. The probes below run them in bash with the pin's values
  sourced and here, prefix, record_name, unfinished_name, LASSI_TOOLCHAINS,
  LASSI_SCRATCH, LASSI_JOBS, TMPDIR, TTMLIR_TOOLCHAIN_DIR, and
  TTMLIR_VENV_DIR set, and the script's own source and build directories
  under the prefix (its `<part>_src="$prefix/..."` and
  `<part>_build="$prefix/..."` lines); pinned_record runs no host tool.
- The host values (task P5.15; plans/local-plan.md section 3.3, item 2):
  the script takes its allowed root from LASSI_ALLOWED_ROOT and its stop
  line from LASSI_STOP_GIB (a positive integer of GiB), both required with
  no default and checked with the other values before the core limit. The
  host profile (tools/local/host-profile.sh) sets them on the workstation;
  on alpha01 the rx command line gives LASSI_ALLOWED_ROOT=/mnt/nvme10/joseph_ufl
  and LASSI_STOP_GIB=145, with which the script keeps and refuses exactly
  what the literal root and 145 GiB did. LASSI_TOOLCHAINS, LASSI_SCRATCH, and
  HOME must each resolve to the allowed root or under it, which is matched
  literally, and a relative allowed root or "/" is refused. The whole-script
  probes run the script with stand-ins for prlimit and for every tool that
  writes or fetches, each an exported bash function that logs its call and
  fails, so the script stops at its first step past its checks and writes
  nothing; the stop line values other than 145 are SYNTHETIC.

No test fetches from the network, configures, or builds anything, and no
test runs a program that could open a device. Fixture texts are SYNTHETIC.
No value here is a measurement: commits, hashes, options, and line numbers
come from upstream source read with `gh api` on 2026-10-12; host tool
versions from the P5.1 spike's read-only probes (rx ids cited); space
figures are the spike's PROJECTED upper ends.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from lassi.toolchains.pins import prefix_pin_name, read_pin

REPO = Path(__file__).resolve().parents[2]
TOOLCHAINS = REPO / "toolchains"
BIBLE = REPO / "docs" / "BIBLE.md"
GATE_CONFIG = REPO / "tools" / "server" / "config.default.json"
PIN_NAME = "tt-mlir"
REQUIREMENTS_NAME = "tt-mlir-requirements.txt"
TRACKED_FETCHED_NAME = "tt-mlir-fetched-sources.txt"
TREE_FETCHED_NAME = "lassi-fetched-sources.txt"
RECORD_NAME = "lassi-install.txt"
UNFINISHED_NAME = "lassi-install.unfinished"
# The bash on PATH, resolved here: on Windows a bare "bash" given to subprocess can start the WSL launcher in
# System32 instead, which sees neither this environment nor Windows paths.
BASH = shutil.which("bash")
needs_bash = pytest.mark.skipif(BASH is None, reason="bash is not available")

# Upstream values, read with `gh api` on 2026-10-12.
# The release tag 0.9.0.dev20260221 is tag object f305713d, whose object is this commit (git/tags API).
TT_MLIR_COMMIT = "5f396fd6ef85780bf1b6b85bef9c7d2f84a85f3c"
TT_MLIR_RELEASE = "0.9.0.dev20260221"
TT_MLIR_URL = "https://github.com/tenstorrent/tt-mlir.git"
# tt-mlir env/CMakeLists.txt at the pin: FLATBUFFERS_VERSION (:4), LLVM_PROJECT_VERSION (:5), and the repositories
# (:41, :72).
FLATBUFFERS_COMMIT = "fb9afbafc7dfe226b9db54d4923bfb8839635274"
FLATBUFFERS_URL = "https://github.com/google/flatbuffers.git"
LLVM_COMMIT = "4efe170d858eb54432f520abb4e7f0086236748b"
LLVM_URL = "https://github.com/llvm/llvm-project.git"
# The patch env/CMakeLists.txt:55 applies to llvm-project: the contents API's blob sha at the pin, equal to
# `git hash-object` of the raw file, and the sha256 of its 1089 bytes.
LLVM_PATCH = "env/patches/affine-allow-symbol-vars.patch"
LLVM_PATCH_BLOB = "2ffdae14d2bcb8244189d8eb044dd093642fa436"
LLVM_PATCH_SHA256 = "61e8ae545e956551b87dc7fb63067d570035becbcf0636ce81ca0971e30f3aac"
# cmake/modules/TTMLIRVersion.cmake:10-14 and :44-49 describe the newest tag matching v[0-9]*.[0-9]*, and :60
# needs one; tt-mlir's only such tag is v0.0 (matching-refs API), 2995 commits before the pin (compare API).
VERSION_TAG = "v0.0"
VERSION_TAG_COMMIT = "a637748e5521f78546027114fabf2023be63901a"
# What ttmlir-opt --version prints: tools/ttmlir-opt/ttmlir-opt.cpp:21-22 calls MlirOptMain and adds no version
# printer, so LLVM's prints "<PACKAGE_NAME> version <PACKAGE_VERSION>" (llvm/lib/Support/CommandLine.cpp:2565-2570
# at LLVM_COMMIT), PACKAGE_NAME LLVM (llvm/CMakeLists.txt:423) and PACKAGE_VERSION 22.0.0git
# (llvm/CMakeLists.txt:24-25; cmake/Modules/LLVMVersion.cmake:3-14).
LLVM_VERSION_LINE = "LLVM version 22.0.0git"
# The options tt-mlir's CMakeLists.txt defines at the pin, by line: option() at :32-48 and :59-60, and the cache
# variable TTMLIR_ENABLE_BINDINGS_PYTHON at :84. A -D name outside these is a typo CMake would ignore.
TT_MLIR_OPTIONS = {
    "TT_RUNTIME_ENABLE_PERF_TRACE": 32,
    "TTMLIR_ENABLE_RUNTIME": 33,
    "TTMLIR_ENABLE_PYKERNEL": 34,
    "TTMLIR_ENABLE_TTNN_JIT": 35,
    "TTMLIR_ENABLE_TTRT": 36,
    "TTMLIR_ENABLE_STABLEHLO": 37,
    "TTMLIR_ENABLE_TTIRTONVVM": 38,
    "TTMLIR_ENABLE_OPMODEL": 39,
    "TTMLIR_ENABLE_OPMODEL_TESTS": 40,
    "TTMLIR_ENABLE_SHARED_LIB": 41,
    "TTMLIR_ENABLE_EXPLORER": 42,
    "TTMLIR_ENABLE_TESTS": 43,
    "TTMLIR_ENABLE_TOOLS": 44,
    "TTMLIR_ENABLE_ALCHEMIST": 45,
    "TTMLIR_ENABLE_ALCHEMIST_WHEEL": 46,
    "CODE_COVERAGE": 47,
    "TT_USE_SYSTEM_SFPI": 48,
    "TTMLIR_ENABLE_RUNTIME_TESTS": 59,
    "TT_RUNTIME_DEBUG": 60,
    "TTMLIR_ENABLE_BINDINGS_PYTHON": 84,
}
# Off for P5.4: the runtime (:33) and OpModel (:39), either of which fetches and builds tt-metal
# (third_party/CMakeLists.txt:87, :103-131); the bindings (:84; python/ at :200-204 and the builder, explorer, and
# ttrt tools at tools/CMakeLists.txt:6-27); the tests (:43; test/ at :205-207); the explorer (:42), ttrt (:36), and
# alchemist (:45; tools/CMakeLists.txt:35-37).
OFF_OPTIONS = (
    "TTMLIR_ENABLE_RUNTIME",
    "TTMLIR_ENABLE_OPMODEL",
    "TTMLIR_ENABLE_BINDINGS_PYTHON",
    "TTMLIR_ENABLE_TESTS",
    "TTMLIR_ENABLE_EXPLORER",
    "TTMLIR_ENABLE_TTRT",
    "TTMLIR_ENABLE_ALCHEMIST",
)
# ttmlir-opt and ttmlir-translate build under TTMLIR_ENABLE_TOOLS (tools/CMakeLists.txt:1-4, :29-32), as the targets
# add_llvm_executable names (tools/ttmlir-opt/CMakeLists.txt:1, tools/ttmlir-translate/CMakeLists.txt:1).
TARGETS = ["ttmlir-opt", "ttmlir-translate"]
# Host tools on alpha01 (plans/spikes/p5-mlir-pins.md, Results 4; rx 20261006-214920-exec-4a93): CMake by path
# (/usr/local/bin/cmake first on PATH), Ninja, clang-17 (tt-mlir's CI compiler), its lld, and Python 3.10.
CMAKE_CHOICES = {("/usr/local/bin/cmake", "cmake version 3.31.6"), ("/usr/bin/cmake", "cmake version 4.2.1")}
# Space, PROJECTED (plans/spikes/p5-mlir-pins.md, Results 5, configuration A, upper ends, in hundredths of a GiB):
# the tt-mlir stack and both stacks together, which the bible's MLIR stacks bullet holds under the 145G stop line.
GIB_KIB = 1024 * 1024
TT_MLIR_STACK_HUNDREDTHS = 2330
BOTH_STACKS_HUNDREDTHS = 4245
STOP_KIB = 145 * GIB_KIB
# SYNTHETIC values for the probes' mismatches.
SYNTHETIC_COMMIT = "0123456789abcdef0123456789abcdef01234567"
SYNTHETIC_SHA256 = "ab" * 32
IDENTITY = {
    "GIT_AUTHOR_NAME": "Pin Test",
    "GIT_AUTHOR_EMAIL": "pin-test@example.invalid",
    "GIT_COMMITTER_NAME": "Pin Test",
    "GIT_COMMITTER_EMAIL": "pin-test@example.invalid",
}

FUNCTION = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\(\) \{\n.*?^\}\n", re.S | re.M)
SETTING = re.compile(r"-D([A-Za-z0-9_]+)(?::[A-Z]+)?=(\S+)")
# A command that downloads: git fetch, clone, or pull (directly or through a git_in wrapper), pip install or
# download, curl, or wget.
DOWNLOAD = re.compile(
    r"\bgit(?:_in)?\b.*\b(?:fetch|clone|pull)\b|\bpip3?\b.*\b(?:install|download)\b|\bcurl\b|\bwget\b"
)
# One requirement of the tracked requirements file, after continuation lines are joined: name==version, an optional
# environment marker, and one or more sha256 hashes.
REQUIREMENT = re.compile(
    r"([A-Za-z0-9][A-Za-z0-9._-]*)==([A-Za-z0-9.+!-]+)(?:\s*;[^;]*?)?((?:\s+--hash=sha256:[0-9a-f]{64})+)"
)
# One entry of a recorded fetched-sources list, in toolchains/tt-metal-cpm-sources.txt's two forms.
FETCHED_ENTRY = re.compile(r"\S+ (?:git [0-9a-f]{40} changed=\d+|files-sha256 [0-9a-f]{64})")


# Helpers.


def pin() -> dict[str, str]:
    """Return toolchains/tt-mlir.pin as the repository's pin reader reads it."""
    return read_pin(PIN_NAME)


def script_text() -> str:
    """Return the text of toolchains/tt-mlir.sh."""
    return (TOOLCHAINS / "tt-mlir.sh").read_text(encoding="utf-8")


def sha(data: bytes) -> str:
    """Return the sha256 hex digest of `data`."""
    return hashlib.sha256(data).hexdigest()


def git_blob(data: bytes) -> str:
    """Return the git blob id of `data`, as `git hash-object` computes it."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


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


def requirements() -> dict[str, tuple[str, list[str]]]:
    """Return the tracked requirements file as normalized name -> (version, sha256 list); fail on any other line."""
    text = (TOOLCHAINS / REQUIREMENTS_NAME).read_bytes().decode("ascii")
    joined = re.sub(r"\\\n\s*", " ", text)
    found: dict[str, tuple[str, list[str]]] = {}
    for line in (raw.strip() for raw in joined.splitlines()):
        if not line or line.startswith("#"):
            continue
        match = REQUIREMENT.fullmatch(line)
        assert match, f"{REQUIREMENTS_NAME}: not a name==version line with sha256 hashes: {line!r}"
        name = re.sub(r"[-_.]+", "-", match.group(1)).lower()
        assert name not in found, f"{REQUIREMENTS_NAME} lists {name} twice"
        found[name] = (match.group(2), re.findall(r"sha256:([0-9a-f]{64})", match.group(3)))
    return found


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
    """Run `body` in strict bash in `cwd`, after the named functions of toolchains/tt-mlir.sh."""
    text = script_text()
    return run_probe("set -euo pipefail\n" + "".join(bash_function(text, name) for name in names) + body, cwd, env)


# The globals the whole-script probes set, after the pin's own values (module docstring).
GLOBALS = """set -euo pipefail
here="$PWD/pins"
source "$here/tt-mlir.pin"
LASSI_TOOLCHAINS="$PWD/tc"
LASSI_SCRATCH="$PWD"
LASSI_JOBS=2
TMPDIR="$PWD/tmp"
prefix="$LASSI_TOOLCHAINS/$PREFIX_NAME"
record_name=lassi-install.txt
unfinished_name=lassi-install.unfinished
TTMLIR_TOOLCHAIN_DIR="$prefix/$TOOLCHAIN_DIR"
TTMLIR_VENV_DIR="$prefix/$VENV_DIR"
"""
# The script's source and build directories under the prefix, which the settings functions name (llvm_src for LLD's
# headers, llvm_build for LLVM_DIR, MLIR_DIR, and LLD_DIR): its six `<part>_src=` and `<part>_build=` lines.
LAYOUT = re.compile(r'^[a-z]+_(?:src|build)="\$prefix/[^"\n]+"$', re.M)


def run_with_script(body: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run `body` after GLOBALS, the script's layout lines, and its functions, with the pin's files copied to pins/."""
    (cwd / "pins").mkdir(exist_ok=True)
    for name in ("tt-mlir.pin", REQUIREMENTS_NAME, TRACKED_FETCHED_NAME):
        shutil.copyfile(TOOLCHAINS / name, cwd / "pins" / name)
    text = script_text()
    layout = LAYOUT.findall(text)
    assert len(layout) == 6, layout
    return run_probe(GLOBALS + "".join(f"{line}\n" for line in layout) + "".join(functions(text).values()) + body, cwd)


# The pin file.


def test_pin_parses_and_names_its_prefix() -> None:
    """The pin reads through lassi.toolchains.pins and names its install prefix <NAME>@<VERSION>."""
    values = pin()
    assert values["NAME"] == PIN_NAME
    assert values["VERSION"] in (TT_MLIR_RELEASE, TT_MLIR_COMMIT[:8]), "the release name or the commit's prefix"
    assert values["PREFIX_NAME"] == f"{PIN_NAME}@{values['VERSION']}"
    assert prefix_pin_name(values["PREFIX_NAME"]) == PIN_NAME


def test_pin_holds_the_joint_pin_commit() -> None:
    """COMMIT is the joint pin's tt-mlir commit, the release 0.9.0.dev20260221 (bible Toolchain Pins)."""
    values = pin()
    assert values["COMMIT"] == TT_MLIR_COMMIT
    assert values["URL"] == TT_MLIR_URL


def test_pin_holds_llvm_flatbuffers_and_the_affine_patch() -> None:
    """The two parts env/CMakeLists.txt builds that tt-mlir needs, and the patch, by commit, blob, and sha256."""
    values = pin()
    assert (values["LLVM_COMMIT"], values["LLVM_URL"]) == (LLVM_COMMIT, LLVM_URL)
    assert (values["FLATBUFFERS_COMMIT"], values["FLATBUFFERS_URL"]) == (FLATBUFFERS_COMMIT, FLATBUFFERS_URL)
    assert values["LLVM_PATCH"] == LLVM_PATCH
    assert (values["LLVM_PATCH_BLOB"], values["LLVM_PATCH_SHA256"]) == (LLVM_PATCH_BLOB, LLVM_PATCH_SHA256)


def test_pin_records_the_version_tag_the_configure_describes() -> None:
    """The clone keeps its history, so TTMLIRVersion.cmake finds v0.0 locally and fetches no tags."""
    values = pin()
    assert (values["VERSION_TAG"], values["VERSION_TAG_COMMIT"]) == (VERSION_TAG, VERSION_TAG_COMMIT)


def test_pin_turns_off_every_part_but_the_tools() -> None:
    """The runtime, OpModel, bindings, tests, explorer, ttrt, and alchemist are off; the tools stay on."""
    settings = cmake_settings(pin()["CMAKE_FLAGS"])
    for name in OFF_OPTIONS:
        assert settings.get(name) == "OFF", f"{name} (CMakeLists.txt:{TT_MLIR_OPTIONS[name]}) must be OFF"
    assert settings.get("TTMLIR_ENABLE_TOOLS", "ON") == "ON", "ttmlir-opt and ttmlir-translate need the tools"
    for name, value in settings.items():
        if name.startswith("TTMLIR_ENABLE_") and name != "TTMLIR_ENABLE_TOOLS":
            assert value == "OFF", f"{name} is on"
    assert settings.get("CMAKE_BUILD_TYPE") in ("Release", "MinSizeRel"), "tt-mlir's build type is recorded"
    assert "CMAKE_INSTALL_PREFIX" not in settings, "the script places the build; the pin names no path"


def test_pin_names_only_options_tt_mlir_defines_at_the_pin() -> None:
    """A misspelled option would be ignored by CMake and leave a part on; every tt-mlir name is one it defines."""
    settings = cmake_settings(pin()["CMAKE_FLAGS"])
    prefixes = ("TTMLIR_", "TT_RUNTIME_", "TT_USE_", "CODE_COVERAGE")
    ours = {name for name in settings if name.startswith(prefixes)}
    assert ours <= set(TT_MLIR_OPTIONS), sorted(ours - set(TT_MLIR_OPTIONS))


def test_pin_builds_only_ttmlir_opt_and_ttmlir_translate() -> None:
    """Only the two tools P5 calls are built, not all or install."""
    assert pin()["TARGETS"].split() == TARGETS


def test_pin_builds_llvm_with_mlir_and_lld_for_the_host_target() -> None:
    """LLVM 4efe170d as env/CMakeLists.txt:58-69 configures it, minus the bindings and tests, for the host only."""
    settings = cmake_settings(pin()["LLVM_CMAKE_FLAGS"])
    # The P5 plan's "no tests" and the spike's configuration A; MLIR's and LLD's tests follow LLVM_INCLUDE_TESTS
    # (mlir/CMakeLists.txt:160-162, lld/CMakeLists.txt:193-200 at LLVM_COMMIT). FileCheck, not, count, and llvm-lit
    # come from LLVM_INCLUDE_UTILS (llvm/CMakeLists.txt:1332-1336, :1425-1427), which must stay on.
    assert settings["LLVM_INCLUDE_TESTS"] == "OFF", "upstream leaves LLVM's default, ON (llvm/CMakeLists.txt:864)"
    assert settings.get("LLVM_INCLUDE_UTILS", "ON") == "ON"
    # env/CMakeLists.txt:64 (LIST_SEPARATOR "," at :71 makes it a CMake list); no clang (spike, Results 2).
    assert settings["LLVM_ENABLE_PROJECTS"].split(";") == ["mlir", "lld"]
    # tt-mlir links the host's X86 libraries alone (lib/Target/LLVM/CMakeLists.txt:11-16).
    assert settings["LLVM_TARGETS_TO_BUILD"] in ("host", "X86")
    # The recorded choice: Release (the P5 plan) or MinSizeRel (env/CMakeLists.txt:8, the environment's default).
    assert settings["CMAKE_BUILD_TYPE"] in ("Release", "MinSizeRel")
    assert settings["LLVM_ENABLE_ASSERTIONS"] == "ON", "upstream's setting (env/CMakeLists.txt:67) is kept"
    assert settings["MLIR_ENABLE_BINDINGS_PYTHON"] == "OFF", "the bindings are off (upstream ON at :68)"
    assert "CMAKE_INSTALL_PREFIX" not in settings, "no install to /usr/local (LLVM's default prefix)"


def test_pin_builds_flatbuffers_as_upstream_does() -> None:
    """flatbuffers' settings of env/CMakeLists.txt:37 and :39; the script installs it into TTMLIR_TOOLCHAIN_DIR."""
    settings = cmake_settings(pin()["FLATBUFFERS_CMAKE_FLAGS"])
    assert settings["CMAKE_BUILD_TYPE"] == "Release"
    assert settings["FLATBUFFERS_BUILD_TESTS"] == "OFF"
    assert "CMAKE_INSTALL_PREFIX" not in settings, "the script sets TTMLIR_TOOLCHAIN_DIR as the prefix"


@pytest.mark.parametrize("key", ["CMAKE_FLAGS", "LLVM_CMAKE_FLAGS", "FLATBUFFERS_CMAKE_FLAGS"])
def test_pin_flags_name_only_the_pinned_compilers(key: str) -> None:
    """A compiler a flags value names is clang-17 or clang++-17 (tt-mlir's CI compiler; CMakeLists.txt:3-8)."""
    settings = cmake_settings(pin()[key])
    for name, expected in (("CMAKE_C_COMPILER", "clang-17"), ("CMAKE_CXX_COMPILER", "clang++-17")):
        if name in settings:
            assert settings[name].rsplit("/", 1)[-1] == expected, f"{key} names {settings[name]} as {name}"


def test_pin_names_each_host_tool_with_its_version() -> None:
    """CMake by path, Ninja, clang-17, its lld, and Python 3.10, each with the version the spike's probes read."""
    values = pin()
    assert (values["CMAKE"], values["CMAKE_EXPECT"]) in CMAKE_CHOICES
    assert (values["NINJA"], values["NINJA_EXPECT"]) == ("/usr/bin/ninja", "1.10.1")
    assert (values["CLANG_C"], values["CLANG_CXX"]) == ("clang-17", "clang++-17")
    assert "clang version 17.0.6" in values["CLANG_EXPECT"]
    # tt-mlir links with ld.lld-<clang major> when it finds it (CMakeLists.txt:110-125).
    assert values["LINKER"] == "ld.lld-17" and "LLD 17.0.6" in values["LINKER_EXPECT"]
    # The bindings-off build runs Python only for code generation; 3.10 has distutils (spike, Results 2).
    assert (values["PYTHON"], values["PYTHON_EXPECT"]) == ("/usr/bin/python3.10", "Python 3.10.12")


def test_pin_names_the_installed_tools_and_their_version() -> None:
    """ttmlir-opt and ttmlir-translate by path inside the prefix; EXPECT_VERSION is LLVM's banner line."""
    values = pin()
    for key, tool in (("TTMLIR_OPT", "ttmlir-opt"), ("TTMLIR_TRANSLATE", "ttmlir-translate")):
        path = values[key]
        assert not path.startswith("/") and ".." not in path.split("/"), f"{key} lies inside the prefix"
        assert path.endswith(f"bin/{tool}"), "add_llvm_executable puts it in the build's bin (FindMLIR.cmake:7)"
    assert LLVM_VERSION_LINE in values["EXPECT_VERSION"]


def test_pin_places_the_toolchain_dir_and_venv_under_the_prefix() -> None:
    """TTMLIR_TOOLCHAIN_DIR (env/activate:1-4 defaults it to /opt/ttmlir-toolchain) and the venv lie in the prefix."""
    values = pin()
    for key in ("TOOLCHAIN_DIR", "VENV_DIR"):
        path = values[key]
        assert path and not path.startswith("/") and ".." not in path.split("/"), f"{key} is relative to the prefix"
    assert values["PYTHON_REQUIREMENTS"] == REQUIREMENTS_NAME


def test_pin_plans_space_for_both_stacks_together() -> None:
    """INSTALL_KIB covers the tt-mlir stack's projected upper end, and with RESERVE_KIB both stacks' (PROJECTED)."""
    values = pin()
    install, reserve = values["INSTALL_KIB"], values["RESERVE_KIB"]
    assert install.isdigit() and reserve.isdigit()
    assert int(install) >= kib_of(TT_MLIR_STACK_HUNDREDTHS), "23.30 GiB, the tt-mlir stack (configuration A)"
    assert int(install) + int(reserve) >= kib_of(BOTH_STACKS_HUNDREDTHS), "42.45 GiB, both stacks (configuration A)"


def section(text: str, heading: str) -> str:
    """Return the body of the markdown section `heading` up to the next heading of any level."""
    start = text.index(f"\n{heading}\n")
    end = text.find("\n#", start + len(heading) + 2)
    return text[start : end if end != -1 else len(text)]


def test_pin_values_match_the_bible_mlir_stacks() -> None:
    """The commits and the patch blob are the ones the bible's Toolchain Pins section records."""
    pins = section(BIBLE.read_text(encoding="utf-8"), "### Toolchain Pins")
    values = pin()
    for key in ("COMMIT", "LLVM_COMMIT", "FLATBUFFERS_COMMIT", "LLVM_PATCH_BLOB"):
        assert values[key] in pins, key
    assert values["LLVM_PATCH"].rsplit("/", 1)[-1] in pins


@needs_bash
def test_bash_reads_the_pin_as_read_pin_does(tmp_path: Path) -> None:
    """The shell that sources the pin sees the values read_pin gives (no unquoted ; or # in a value)."""
    values = pin()
    shutil.copy(TOOLCHAINS / "tt-mlir.pin", tmp_path / "tt-mlir.pin")
    body = "set -euo pipefail\nsource ./tt-mlir.pin\n" + "".join(f'printf "%s\\n" "${key}"\n' for key in values)
    result = run_probe(body, tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == list(values.values())


# The tracked lists.


def test_requirements_pin_every_package_with_a_sha256() -> None:
    """Every download pip makes for the venv is pinned by version and checked by sha256 (--require-hashes)."""
    found = requirements()
    assert found
    for name, (_, hashes) in found.items():
        assert hashes, f"{name} has no --hash=sha256"


def test_requirements_hold_sphinx_for_python_3_10_and_the_markdown_builder() -> None:
    """docs/CMakeLists.txt:8-33 needs sphinx-build and sphinx-apidoc; PyPI lists sphinx 8.2.0 as needing Python 3.11."""
    found = requirements()
    assert "sphinx-markdown-builder" in found
    major, minor = (int(part) for part in found["sphinx"][0].split(".")[:2])
    assert (major, minor) <= (8, 1), "sphinx 8.1.3 requires Python >=3.10 and 8.2.0 >=3.11 (PyPI, 2026-10-12)"


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


# The files.


@pytest.mark.parametrize("name", ["tt-mlir.pin", "tt-mlir.sh", REQUIREMENTS_NAME, TRACKED_FETCHED_NAME])
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
    for name in ("tt-mlir.pin", "tt-mlir.sh", REQUIREMENTS_NAME, TRACKED_FETCHED_NAME):
        text = (TOOLCHAINS / name).read_text(encoding="utf-8")
        for pattern in patterns:
            assert not re.search(pattern, text), f"{name} matches the gate pattern {pattern}"


@needs_bash
def test_script_parses_as_bash() -> None:
    """bash -n accepts the script."""
    assert (TOOLCHAINS / "tt-mlir.sh").is_file()
    assert BASH is not None
    # A relative POSIX path, since bash on Windows would read backslashes in a native path as escapes.
    result = subprocess.run([BASH, "-n", "toolchains/tt-mlir.sh"], capture_output=True, text=True, cwd=REPO)
    assert result.returncode == 0, result.stderr


# The script, read as text: where it may write and what it never runs.


def test_script_is_strict_sources_its_pin_and_needs_the_gate_environment() -> None:
    """The tt-metal.sh preamble, with HOME also required to lie in the scratch root and a 1-byte core limit."""
    text = script_text()
    assert text.startswith("#!/usr/bin/env bash\n")
    assert "set -euo pipefail" in text
    # The core limit is prlimit's 1 byte, not `ulimit -c 0` (test_script_holds_the_core_limit_at_one_byte_first).
    assert CORE_LIMIT in text and CORE_READ in text
    # The allowed root and the stop line come from the host profile, or on alpha01 the command line, with no default.
    for name in ("LASSI_TOOLCHAINS", "LASSI_SCRATCH", "LASSI_JOBS", "LASSI_ALLOWED_ROOT", "LASSI_STOP_GIB"):
        assert f'"${{{name}:?' in text, name
    loop = [line for line in logic_lines(text) if line.startswith("for root in ")]
    assert len(loop) == 1, loop
    for name in ("$LASSI_TOOLCHAINS", "$LASSI_SCRATCH", "$HOME"):
        assert f'"{name}"' in loop[0], f"{name} must lie in the allowed root (Agent Rule 7)"
    assert ROOT_CASE.search(text), "the root check's case pattern is the quoted allowed root, then /*"
    assert 'export TMPDIR="$LASSI_SCRATCH/tmp"' in text
    assert 'prefix="$LASSI_TOOLCHAINS/$PREFIX_NAME"' in text


# The allowed root and the stop line (task P5.15).


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
# The root check's case line: the resolved root and a slash, matched against the allowed root, quoted so that a glob
# character in it is matched as itself, and anything under it.
ROOT_CASE = re.compile(r'case "\$\(realpath -m "\$root"\)/" in\n +"\$\{?[A-Za-z_][A-Za-z0-9_]*\}?"/\*\) ;;')


def run_script(cwd: Path, values: dict[str, str]) -> tuple[subprocess.CompletedProcess[str], list[str], str]:
    """Run toolchains/tt-mlir.sh from `cwd` with only `values` of SCRIPT_VARIABLES (and HOME) set, under stand-ins.

    Each value is a double-quoted bash word, so "$PWD" in it names `cwd`. Each stand-in is an exported bash
    function that logs its call to calls.log and fails. Returns the result, the logged calls, and `cwd` as bash
    spells it.
    """
    lines = ["set -euo pipefail", 'printf "%s" "$PWD" >pwd.txt', ": >calls.log", 'export CALLS_LOG="$PWD/calls.log"']
    for tool in STAND_INS:
        lines += [f'{tool}() {{ echo "{tool} $*" >>"$CALLS_LOG"; return 97; }}', f"export -f {tool}"]
    lines.append("unset " + " ".join(SCRIPT_VARIABLES))
    lines += [f'export {key}="{value}"' for key, value in values.items()]
    lines.append(f"exec \"$BASH\" '{(TOOLCHAINS / 'tt-mlir.sh').as_posix()}'")
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
def test_script_keeps_and_refuses_with_alpha01s_values_as_before(
    tmp_path: Path, changed: dict[str, str], kept: bool
) -> None:
    """With alpha01's root and 145 on its command line, the roots and HOME the literal root kept are kept, no more."""
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
@pytest.mark.parametrize("value", ["0", "abc", "-1", "1.5", "145G"])
def test_script_refuses_a_stop_line_that_is_not_a_positive_integer(tmp_path: Path, value: str) -> None:
    """LASSI_STOP_GIB is checked with the other values, before the core limit or any tool, and the message prints it."""
    result, calls, _ = run_script(tmp_path, {**ALPHA01, "LASSI_STOP_GIB": value})
    stderr = refused(result, calls)
    assert "LASSI_STOP_GIB" in stderr and f"'{value}'" in stderr, stderr


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


# The core limit (Agent Rule 7). alpha01's core handler pipes a core to systemd-coredump, which ignores a limit of 0
# (`ulimit -c 0`) and stores the core on the root filesystem; the kernel aborts a piped dump only at exactly 1 byte
# (plans/spikes/p0-sandbox-hardening.md, R6, rx 20260923-133451-exec-7c3d). bash's `ulimit -c` counts 1024-byte
# blocks, so prlimit sets it and getrlimit reads it back (plans/LESSONS.md; plans/runs/p17-gate-a/run.sh:73-75, :227).
CORE_LIMIT = "prlimit --pid $$ --core=1:1"
CORE_READ = "resource.getrlimit(resource.RLIMIT_CORE) != (1, 1)"
# A code line that runs a tool: a pinned host tool by its variable, or git, pip, CMake, Ninja, a download, a new
# directory or temporary file, readelf, or find.
TOOL_RUN = re.compile(
    r'"\$(?:CMAKE|NINJA|PYTHON|CLANG_C|CLANG_CXX|LINKER)"'
    r"|\b(?:git|git_in|pip3?|cmake|ninja|curl|wget|mkdir|mktemp|readelf|find)\b"
)


def core_block(text: str) -> str:
    """Return the script's core-limit block, from its `if ! prlimit` line through its `fi` line."""
    start = text.index(f"\nif ! {CORE_LIMIT}") + 1
    return text[start : text.index("\nfi\n", start) + len("\nfi\n")]


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
    assert start < main_start + call_position(main_section(text), {"check_host"})


@needs_bash
@pytest.mark.parametrize(
    ("set_status", "read_status", "holds"),
    [(0, 0, True), (1, 0, False), (0, 1, False)],
    ids=["held", "prlimit-fails", "reads-otherwise"],
)
def test_core_limit_block_refuses_unless_set_and_read_back(
    tmp_path: Path, set_status: int, read_status: int, holds: bool
) -> None:
    """The block exits 2 with one tt-mlir line when prlimit fails or the limit does not read (1, 1)."""
    stubs = f"prlimit() {{ return {set_status}; }}\nread_back() {{ return {read_status}; }}\nPYTHON=read_back\n"
    result = run_probe("set -euo pipefail\n" + stubs + core_block(script_text()) + "echo held\n", tmp_path)
    if holds:
        assert (result.returncode, result.stdout) == (0, "held\n"), result.stdout + result.stderr
        return
    assert result.returncode == 2 and result.stdout == "", result.stdout + result.stderr
    lines = result.stderr.splitlines()
    assert len(lines) == 1 and lines[0].startswith("tt-mlir:") and "1 byte" in lines[0], result.stderr


def test_script_never_runs_the_environment_build_or_activate() -> None:
    """No env/CMakeLists.txt, env/activate, venv script, tt-metal debugger, unpinned pip install, or sudo."""
    lines = logic_lines(script_text())
    forbidden = ("env/activate", "env/CMakeLists", "init_venv", "install-tt-triage", "install_debugger",
                 "mlir/python/requirements", "nanobind", "sudo")
    for word in forbidden:
        assert not [line for line in lines if word in line], word
    sourced = [line for line in lines if re.match(r"(source|\.)\s", line)]
    assert sourced == ['source "$here/tt-mlir.pin"'], "the pin is the only file the script sources"
    assert not [line for line in lines if re.search(r'-S\s+"?[^\s"]*/env/?"?(\s|$)', line)], "no configure of env/"


def test_script_writes_no_system_path() -> None:
    """Only the root-filesystem check names /tmp, /var/tmp, or /opt; no command uses a /usr/, /etc/, or /opt/ path."""
    shebang, _, rest = script_text().partition("\n")
    assert shebang == "#!/usr/bin/env bash"
    commands = "\n".join(logic_lines(rest))
    for system_path in ("/usr/", "/opt/", "/etc/"):
        assert system_path not in commands, system_path
    naming = [line for line in logic_lines(rest) if re.search(r'(^|[\s"=(])/((var/)?tmp|opt)\b', line)]
    assert naming == ['check_root_fs "$marker" /tmp /var/tmp /opt']


def test_root_fs_check_is_the_one_tt_metal_sh_uses() -> None:
    """The P4.2 method: the same find over the named directories, read the same way (messages aside)."""
    ours = logic_lines(bash_function(script_text(), "check_root_fs"))
    theirs = logic_lines(bash_function((TOOLCHAINS / "tt-metal.sh").read_text(encoding="utf-8"), "check_root_fs"))
    assert ours == theirs


def test_script_clears_the_build_environment_before_its_own_variables() -> None:
    """No compiler, search-path, Python, tt-metal, or tt-mlir variable of the caller reaches the build."""
    text = script_text()
    unset = [line for line in logic_lines(text) if line.startswith("unset CC CXX ")]
    assert len(unset) == 1, unset
    names = set(unset[0].split()) - {"unset"}
    required = {"CC", "CXX", "CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS", "CPATH", "C_INCLUDE_PATH",
                "CPLUS_INCLUDE_PATH", "LIBRARY_PATH", "LD_LIBRARY_PATH", "CMAKE_PREFIX_PATH", "PKG_CONFIG_PATH",
                "PYTHONPATH", "PYTHONHOME"}
    assert required <= names, required - names
    loop = text.index("$(compgen -e)")
    pattern = text[loop : text.index("esac", loop)]
    for glob in ("TTMLIR_*", "TT_MLIR_*", "TT_METAL_*"):
        assert glob in pattern, glob
    assert loop < text.index('export TTMLIR_TOOLCHAIN_DIR="$prefix/$TOOLCHAIN_DIR"')


def test_script_sets_the_tt_mlir_environment_under_the_prefix() -> None:
    """The variables env/activate would set, set by the script itself, under the prefix, before any download."""
    text = script_text()
    limit = text.index("\ntrap cleanup EXIT\n") + first_download(text)
    for line in ("export TTMLIR_ENV_ACTIVATED=1", 'export TTMLIR_TOOLCHAIN_DIR="$prefix/$TOOLCHAIN_DIR"',
                 'export TTMLIR_VENV_DIR="$prefix/$VENV_DIR"'):
        assert text.index(line) < limit, line
    paths = [line for line in logic_lines(text) if line.startswith("export PATH=")]
    assert len(paths) == 1, paths
    assert paths[0].startswith('export PATH="$TTMLIR_VENV_DIR/bin:'), "the venv's bin first (docs/CMakeLists.txt:9)"
    assert "$TTMLIR_TOOLCHAIN_DIR/bin" in paths[0], "flatc is found there (BuildFlatbuffers.cmake:1)"
    assignments = [line for line in logic_lines(text) if re.search(r"\bTTMLIR_TOOLCHAIN_DIR=", line)]
    assert len(assignments) == 1, assignments


def test_script_installs_only_into_the_toolchain_dir() -> None:
    """flatbuffers installs into TTMLIR_TOOLCHAIN_DIR, where flatc is checked; nothing installs anywhere else."""
    text = script_text()
    prefixes = [line for line in code_lines(text) if "CMAKE_INSTALL_PREFIX" in line]
    assert prefixes, "flatbuffers is installed into the toolchain dir (env/CMakeLists.txt:38)"
    for line in prefixes:
        assert re.search(r'CMAKE_INSTALL_PREFIX="?\$\{?TTMLIR_TOOLCHAIN_DIR\b', line), line
    assert "$TTMLIR_TOOLCHAIN_DIR/bin/flatc" in text


def test_script_names_cmake_by_path_and_builds_with_the_gate_job_count() -> None:
    """CMake runs as "$CMAKE", Ninja is named to it, and every build runs at LASSI_JOBS."""
    lines = logic_lines(script_text())
    assert not [line for line in lines if re.search(r"(^|[\s;&|(])cmake\s", line)], "no cmake from PATH"
    assert [line for line in lines if '"$CMAKE"' in line]
    assert [line for line in code_lines(script_text()) if "-DCMAKE_MAKE_PROGRAM=" in line and "$NINJA" in line]
    builds = [line for line in lines if "--build" in line]
    assert builds and all('--parallel "$LASSI_JOBS"' in line for line in builds), builds


def test_script_names_the_pinned_compilers_to_each_configure() -> None:
    """clang-17 and clang++-17 by name (tt-mlir defaults to an unversioned clang, CMakeLists.txt:3-8)."""
    values = pin()
    keys = ("CMAKE_FLAGS", "LLVM_CMAKE_FLAGS", "FLATBUFFERS_CMAKE_FLAGS")
    named = [{"CMAKE_C_COMPILER", "CMAKE_CXX_COMPILER"} <= set(cmake_settings(values[key])) for key in keys]
    if all(named):
        return
    lines = code_lines(script_text())
    assert [line for line in lines if re.search(r'-DCMAKE_C_COMPILER="?\$\{?CLANG_C\b', line)]
    assert [line for line in lines if re.search(r'-DCMAKE_CXX_COMPILER="?\$\{?CLANG_CXX\b', line)]


def test_script_builds_the_pinned_targets() -> None:
    """The tt-mlir build names the pin's TARGETS, and nothing runs the install target of LLVM or tt-mlir."""
    text = script_text()
    builders = [name for name, body in functions(text).items() if "--build" in body and "$TARGETS" in body]
    direct = [line for line in logic_lines(main_section(text)) if "--build" in line and "$TARGETS" in line]
    assert builders or direct, "a build names $TARGETS"
    assert not [line for line in logic_lines(text) if re.search(r"--target\s+all\b|\bninja\s+install\b", line)]


def test_script_installs_python_packages_only_with_hashes() -> None:
    """The venv comes from the pinned Python, and pip installs only the tracked requirements, hash-checked."""
    lines = logic_lines(script_text())
    assert [line for line in lines if '"$PYTHON" -m venv' in line]
    pips = [line for line in lines if re.search(r"\bpip3?\b.*\binstall\b", line)]
    assert pips, "the requirements are installed"
    for line in pips:
        assert "--require-hashes" in line and re.search(r"(^|\s)-r\s", line), line
        assert "$PYTHON_REQUIREMENTS" in line or REQUIREMENTS_NAME in line, line


def test_script_checks_the_version_tag_of_its_clone() -> None:
    """The clone's describe gives VERSION_TAG, so the configure needs no tag fetch (TTMLIRVersion.cmake:17-40)."""
    text = "\n".join(logic_lines(script_text()))
    assert "describe --tags" in text
    assert "$VERSION_TAG" in text


def test_script_checks_the_patch_before_applying_it() -> None:
    """Each step that applies tt-mlir's patch to LLVM first checks it with check_patch."""
    text = script_text()
    assert "$LLVM_PATCH" in text
    appliers = {name: body_lines(text, name) for name in functions(text)}
    appliers = {name: lines for name, lines in appliers.items()
                if any(re.search(r"\bgit(?:_in)?\b.*\bapply\b|\bpatch\s+-p", line) for line in lines)}
    assert appliers, "the patch is applied"
    for name, lines in appliers.items():
        checked = [i for i, line in enumerate(lines) if calls(line, {"check_patch"})]
        applied = [i for i, line in enumerate(lines) if re.search(r"\bapply\b|\bpatch\s+-p", line)]
        assert checked and checked[0] < applied[0], f"{name} applies the patch before check_patch"


def test_script_checks_each_configured_cache() -> None:
    """After each configure, check_cache reads the pinned settings back from the CMake cache (runtime off included)."""
    checks = [line for line in logic_lines(script_text()) if calls(line, {"check_cache"})]
    for flags in ('"$CMAKE_FLAGS"', '"$LLVM_CMAKE_FLAGS"'):
        assert [line for line in checks if flags in line], f"no check_cache of {flags}"


def test_script_fetches_and_builds_no_tt_metal() -> None:
    """With the runtime and OpModel off nothing fetches tt-metal; the script names none either."""
    assert not [line for line in logic_lines(script_text()) if "tt-metal" in line]


def test_check_host_checks_every_pinned_tool() -> None:
    """check_host refuses a host tool whose version is not the pinned one."""
    body = "\n".join(body_lines(script_text(), "check_host"))
    for key in ("CMAKE_EXPECT", "NINJA_EXPECT", "CLANG_EXPECT", "LINKER_EXPECT", "PYTHON_EXPECT"):
        assert f'"${key}"' in body, key


def test_check_tools_reads_the_comment_section_of_both_tools() -> None:
    """check_linked runs inside the loop over both tools, so ttmlir-translate's compiler and linker are checked too."""
    lines = body_lines(script_text(), "check_tools")
    loop = lines.index("for key in TTMLIR_OPT TTMLIR_TRANSLATE; do")
    end = lines.index("done", loop)
    linked = [i for i, line in enumerate(lines) if calls(line, {"check_linked"})]
    assert linked and all(loop < i < end and '"$tool"' in lines[i] for i in linked), lines


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
    for step in ("check_prefix", "check_host", "check_space"):
        assert kept < call_position(main, {step}), step
    assert kept < first_download(text)
    verify = "\n".join(body_lines(text, "verify_install"))
    assert "check_fetched_sources" in verify and '"$EXPECT_VERSION"' in verify


def test_every_refusal_comes_before_any_download() -> None:
    """The prefix, host, and space checks run before the first fetch or pip install, and the marker is written there."""
    text = script_text()
    main, download = main_section(text), first_download(text)
    for step in ("check_prefix", "check_host", "check_space"):
        assert call_position(main, {step}) < download, f"{step} runs after a download"
    assert main.index('pinned_record >"$prefix/$unfinished_name"') < download


def test_space_check_plans_for_both_stacks() -> None:
    """The space check's planned KiB is INSTALL_KIB plus RESERVE_KIB (P5.4: both stacks together)."""
    main = main_section(script_text())
    found = [line for _, line in positioned(main) if calls(line, {"check_space"})]
    assert found, "the main steps run check_space"
    line = found[0]
    argument = line.split("check_space", 1)[1]
    if not ("INSTALL_KIB" in argument and "RESERVE_KIB" in argument):
        variable = re.search(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)", argument)
        assert variable, line
        assigned = [ln for _, ln in positioned(main) if ln.startswith(f"{variable.group(1)}=")]
        assert assigned and "INSTALL_KIB" in assigned[0] and "RESERVE_KIB" in assigned[0], line


def test_script_records_compares_builds_verifies_and_records_in_order() -> None:
    """Fetched sources are recorded and compared before the tt-mlir build; the record is written last."""
    text = script_text()
    main = main_section(text)
    builders = {name for name, body in functions(text).items() if "--build" in body and "$TARGETS" in body}
    # Each step is looked for after the one before it, so the kept-install block's calls are not counted.
    recorded = call_position(main, {"record_fetched_sources"}, after=first_download(text))
    compared = call_position(main, {"check_fetched_sources"}, after=recorded)
    built = call_position(main, builders, after=compared) if builders else main.index("$TARGETS", compared)
    version = step_position(text, '"$EXPECT_VERSION"', built)
    checked = main.index('check_root_fs "$marker" /tmp /var/tmp /opt', version)
    assert main.index('mv -T "$prefix/$unfinished_name" "$prefix/$record_name"', checked) > checked
    assert TREE_FETCHED_NAME in text and TRACKED_FETCHED_NAME in text


# Behavior: single functions of the script, run in bash on local files.


SPACE_STUBS = """du() { if [ -n "$DU_KIB" ]; then printf '%s\\t%s\\n' "$DU_KIB" "${@: -1}"; fi; }
df() { printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\\n/dev/x 1 1 %s 1%% /mnt\\n' "$DF_KIB"; }
LASSI_SCRATCH=/mnt/nvme10/scratch
"""


def space_body(stop_gib: int) -> str:
    """Return a check_space probe body with the du and df stand-ins and LASSI_STOP_GIB set to `stop_gib`."""
    return SPACE_STUBS + f'LASSI_STOP_GIB={stop_gib}\ncheck_space "1000"\n'


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
        assert "tt-mlir:" in result.stderr


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
    """The record a kept install is compared with changes with any pinned commit, hash, flag, tool, or Python pin."""
    result = run_with_script("pinned_record >rec1.txt\npinned_record >rec2.txt\n", tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    record = (tmp_path / "rec1.txt").read_text(encoding="ascii")
    assert record == (tmp_path / "rec2.txt").read_text(encoding="ascii"), "the record is the same each time"
    assert record.splitlines()[0].split()[:2] == [PIN_NAME, TT_MLIR_COMMIT], "NAME and COMMIT on the first line"
    values = pin()
    keys = ("URL", "VERSION_TAG", "LLVM_COMMIT", "LLVM_URL", "FLATBUFFERS_COMMIT", "LLVM_PATCH_BLOB",
            "LLVM_PATCH_SHA256", "CMAKE_FLAGS", "LLVM_CMAKE_FLAGS", "FLATBUFFERS_CMAKE_FLAGS", "TARGETS", "CMAKE",
            "CLANG_C", "CLANG_CXX", "LINKER", "PYTHON")
    for key in keys:
        assert values[key] in record, key
    assert sha((TOOLCHAINS / REQUIREMENTS_NAME).read_bytes()) in record, "the Python pins are part of the record"
    assert not (tmp_path / "tc").exists(), "pinned_record writes nothing"


@needs_bash
def test_pinned_record_holds_the_settings_the_script_adds(tmp_path: Path) -> None:
    """Each configure's added -D settings are a record line, as tt-metal.sh's are, the LLD workaround among them."""
    body = "pinned_record >rec.txt\nflatbuffers_settings >fb.txt\nllvm_settings >llvm.txt\nttmlir_settings >tt.txt\n"
    result = run_with_script(body, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    record = {line.split(" ", 1)[0]: line for line in (tmp_path / "rec.txt").read_text(encoding="ascii").splitlines()}
    for key, name in (("flatbuffers-local", "fb.txt"), ("llvm-local", "llvm.txt"), ("local", "tt.txt")):
        settings = (tmp_path / name).read_text(encoding="ascii").splitlines()
        assert settings and all(setting.startswith("-D") for setting in settings), settings
        # tr turns each setting's newline into one space (toolchains/tt-metal.sh, pinned_record).
        assert record.get(key) == key + " " + "".join(f"{setting} " for setting in settings), key
    local = record["local"].split()
    assert [word for word in local if word.startswith("-DLLD_DIR=") and word.endswith("/lib/cmake/lld")]
    assert [word for word in local if word.startswith("-DCMAKE_CXX_FLAGS=-isystem") and word.endswith("/lld/include")]
    for key in ("flatbuffers-local", "llvm-local", "local"):
        assert "-DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF" in record[key].split(), key


def prefix_state(prefix: Path) -> dict[str, bytes]:
    """Return every file under `prefix` by relative path, with its bytes ({} when the prefix does not exist)."""
    if not prefix.exists():
        return {}
    return {path.relative_to(prefix).as_posix(): path.read_bytes() for path in prefix.rglob("*") if path.is_file()}


@needs_bash
@pytest.mark.parametrize(
    ("case", "kept"),
    [("absent", True), ("resume", True), ("other-commit", False), ("other-patch", False), ("unknown", False)],
)
def test_check_prefix_keeps_only_a_fresh_or_resumable_prefix(tmp_path: Path, case: str, kept: bool) -> None:
    """An unfinished tree at the pin resumes; any other install or content at the prefix is refused, never moved."""
    made = run_with_script("pinned_record >rec.txt\n", tmp_path)
    assert made.returncode == 0, made.stdout + made.stderr
    record = (tmp_path / "rec.txt").read_text(encoding="ascii")
    prefix = tmp_path / "tc" / pin()["PREFIX_NAME"]
    contents = {
        "resume": {UNFINISHED_NAME: record},
        "other-commit": {RECORD_NAME: record.replace(TT_MLIR_COMMIT, SYNTHETIC_COMMIT)},
        "other-patch": {UNFINISHED_NAME: record.replace(LLVM_PATCH_SHA256, SYNTHETIC_SHA256)},
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


PATCH = b"# SYNTHETIC patch text for the check_patch probe; not tt-mlir's patch.\n--- a/x\n+++ b/x\n"


@needs_bash
@pytest.mark.parametrize(
    ("written", "blob", "passes"),
    [(PATCH, git_blob(PATCH), True), (PATCH + b"x\n", git_blob(PATCH), False), (PATCH, "0" * 40, False),
     (None, git_blob(PATCH), False)],
    ids=["pinned", "other-bytes", "other-blob", "missing"],
)
def test_check_patch_needs_the_pinned_sha256_and_blob(
    tmp_path: Path, written: bytes | None, blob: str, passes: bool
) -> None:
    """check_patch <file> passes only a file with the pin's sha256 and git blob id."""
    if written is not None:
        (tmp_path / "p.patch").write_bytes(written)
    body = f"LLVM_PATCH_SHA256={sha(PATCH)}\nLLVM_PATCH_BLOB={blob}\ncheck_patch p.patch\n"
    result = run_functions(("check_patch",), body, tmp_path)
    assert (result.returncode == 0) == passes, result.stdout + result.stderr
    if not passes:
        assert "tt-mlir:" in result.stderr


ENTRY_A = "alpha/0001 git " + "1" * 40 + " changed=0"
ENTRY_B = "beta/0002 files-sha256 " + "2" * 64
TRACKED = f"# SYNTHETIC tracked list for the probe\n\n{ENTRY_A}\n{ENTRY_B}\n"


@needs_bash
@pytest.mark.parametrize(
    ("tree", "tracked", "passes"),
    [
        (f"{ENTRY_A}\n{ENTRY_B}  \n", TRACKED, True),
        (f"{ENTRY_A.replace('changed=0', 'changed=1')}\n{ENTRY_B}\n", TRACKED, False),
        (f"{ENTRY_A}\n{ENTRY_B}\n" + "gamma/0003 git " + "3" * 40 + " changed=0\n", TRACKED, False),
        (f"{ENTRY_A}\n", TRACKED, False),
        (None, TRACKED, False),
        (f"{ENTRY_A}\n{ENTRY_B}\n", "# SYNTHETIC tracked list\nPLACEHOLDER\n", False),
    ],
    ids=["same", "other-line", "added", "missing-line", "no-tree-list", "placeholder"],
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
        assert "tt-mlir:" in result.stderr
        assert TRACKED_FETCHED_NAME in result.stderr or TREE_FETCHED_NAME in result.stderr, "the message names a list"
    if "PLACEHOLDER" in tracked:
        assert "PLACEHOLDER" in result.stderr


CACHE_FLAGS = "-G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS=mlir;lld"
CACHE_LINES = ["# SYNTHETIC CMake cache for the probe", "CMAKE_BUILD_TYPE:STRING=Release",
               "LLVM_ENABLE_PROJECTS:STRING=mlir;lld", "CMAKE_GENERATOR:INTERNAL=Ninja"]


@needs_bash
@pytest.mark.parametrize(
    ("lines", "key"),
    [
        (CACHE_LINES, None),
        ([line.replace("=Release", "=MinSizeRel") for line in CACHE_LINES], "CMAKE_BUILD_TYPE"),
        ([line for line in CACHE_LINES if not line.startswith("LLVM_")], "LLVM_ENABLE_PROJECTS"),
        (None, "CMakeCache.txt"),
    ],
    ids=["pinned", "other-value", "missing-setting", "no-cache"],
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
