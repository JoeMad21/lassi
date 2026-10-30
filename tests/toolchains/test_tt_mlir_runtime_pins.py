"""Tests for the tt-mlir runtime pin, its tracked lists, and its install script (task P17.48).

Bible: Toolchain Pins (the joint pin; pin files and install rules), Host Facts
(the scratch cap and stop line), Agent Rules 7, 9, 10, and 17; the owner's
answer to OQ-111 (option (b): a tt-mlir runtime at the pin with MLIR's Python
bindings and ttrt, in a new prefix); plans/spikes/ttir-d2m.md, section 4, and
plans/spikes/ttmlir-runtime.md (the design).

The design: toolchains/tt-mlir-runtime.pin names a build variant of the joint
pin's tt-mlir (BUILD runtime, PREFIX_NAME tt-mlir@0.9.0.dev20260221-runtime)
with the compiler pin's sources, host tools, and layout, and the runtime, the
TTMetal and TTNN runtimes, MLIR's and tt-mlir's Python bindings, and ttrt on.
toolchains/tt-mlir-runtime.sh builds it in place in that prefix: it fetches
its own sources by the same commits, never writes to or reads from the compiler
install (toolchains/tt-mlir.pin's prefix), and refuses a layout in which either
prefix holds the other and an install record while anything there changed. It
installs a venv from a hash-checked requirements file through a wheelhouse,
builds LLVM with MLIR's Python bindings, then tt-mlir's tt-metal through its
ExternalProject's step targets (clone, configure, build), each checked, then
the pin's TARGETS and ttrt, every build a Ninja run at LASSI_JOBS that stops
when the scratch disk's free space falls under LASSI_FREE_FLOOR_GIB. It holds
RLIMIT_CORE at 1 byte, runs at the gate's niceness or more, and opens no
device.

The probes run the script's functions in bash with the pin's values sourced
and stand-ins for the tools they would run; the whole-script probes run it
with stand-ins for prlimit and every tool that writes or fetches, each an
exported bash function that logs its call and fails, so the script stops at
its first step past its checks and writes nothing. No test fetches from the
network, configures, or builds anything, and no test runs a program that could
open a device. Fixture texts and the stop line and floor values other than the
proposed ones are SYNTHETIC. No value here is a measurement: commits, hashes,
options, and line numbers come from upstream source read on 2026-10-29.
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

from lassi.toolchains.pins import prefix_pin_name, read_pin, recorded_version

REPO = Path(__file__).resolve().parents[2]
TOOLCHAINS = REPO / "toolchains"
GATE_CONFIG = REPO / "tools" / "server" / "config.default.json"
PIN_NAME = "tt-mlir-runtime"
SCRIPT_NAME = "tt-mlir-runtime.sh"
REQUIREMENTS_NAME = "tt-mlir-runtime-requirements.txt"
COMPILER_REQUIREMENTS_NAME = "tt-mlir-requirements.txt"
TRACKED_FETCHED_NAME = "tt-mlir-runtime-fetched-sources.txt"
FILES = ("tt-mlir-runtime.pin", SCRIPT_NAME, REQUIREMENTS_NAME, TRACKED_FETCHED_NAME)
RECORD_NAME = "lassi-install.txt"
UNFINISHED_NAME = "lassi-install.unfinished"
BASH = shutil.which("bash")
needs_bash = pytest.mark.skipif(BASH is None, reason="bash is not available")
GIB_KIB = 1024 * 1024

# The option() lines of tt-mlir's CMakeLists.txt at 5f396fd6 (:32-48, :51-53, :59-60) and its cache variable
# TTMLIR_ENABLE_BINDINGS_PYTHON (:84). A -D name outside these is a typo CMake would ignore.
TT_MLIR_OPTIONS = {
    "TT_RUNTIME_ENABLE_PERF_TRACE": 32, "TTMLIR_ENABLE_RUNTIME": 33, "TTMLIR_ENABLE_PYKERNEL": 34,
    "TTMLIR_ENABLE_TTNN_JIT": 35, "TTMLIR_ENABLE_TTRT": 36, "TTMLIR_ENABLE_STABLEHLO": 37,
    "TTMLIR_ENABLE_TTIRTONVVM": 38, "TTMLIR_ENABLE_OPMODEL": 39, "TTMLIR_ENABLE_OPMODEL_TESTS": 40,
    "TTMLIR_ENABLE_SHARED_LIB": 41, "TTMLIR_ENABLE_EXPLORER": 42, "TTMLIR_ENABLE_TESTS": 43,
    "TTMLIR_ENABLE_TOOLS": 44, "TTMLIR_ENABLE_ALCHEMIST": 45, "TTMLIR_ENABLE_ALCHEMIST_WHEEL": 46,
    "CODE_COVERAGE": 47, "TT_USE_SYSTEM_SFPI": 48, "TT_RUNTIME_ENABLE_TTNN": 51, "TT_RUNTIME_ENABLE_TTMETAL": 52,
    "TT_RUNTIME_ENABLE_DISTRIBUTED": 53, "TTMLIR_ENABLE_RUNTIME_TESTS": 59, "TT_RUNTIME_DEBUG": 60,
    "TTMLIR_ENABLE_BINDINGS_PYTHON": 84,
}
# On for option (b): the runtime (:33), the TTMetal runtime that runs TTIR and D2M flatbuffers (:52), the TTNN
# runtime, upstream's default with the runtime (:51), the bindings (:84), and ttrt (:36).
ON_OPTIONS = ("TTMLIR_ENABLE_RUNTIME", "TT_RUNTIME_ENABLE_TTMETAL", "TT_RUNTIME_ENABLE_TTNN",
              "TTMLIR_ENABLE_BINDINGS_PYTHON", "TTMLIR_ENABLE_TTRT")
# Off: the distributed runtime (:53); the system sfpi (:48), since /opt/tenstorrent/sfpi on alpha01 is not the
# pinned sfpi; and what the compiler pin also turns off.
OFF_OPTIONS = ("TT_RUNTIME_ENABLE_DISTRIBUTED", "TT_USE_SYSTEM_SFPI", "TTMLIR_ENABLE_OPMODEL", "TTMLIR_ENABLE_TESTS",
               "TTMLIR_ENABLE_RUNTIME_TESTS", "TTMLIR_ENABLE_EXPLORER", "TTMLIR_ENABLE_ALCHEMIST",
               "TTMLIR_ENABLE_PYKERNEL", "TTMLIR_ENABLE_TTNN_JIT", "TTMLIR_ENABLE_STABLEHLO",
               "TT_RUNTIME_ENABLE_PERF_TRACE", "TT_RUNTIME_DEBUG")
# Values the runtime pin shares with toolchains/tt-mlir.pin: the sources, the host tools, and the compiler layout.
SHARED_KEYS = ("NAME", "VERSION", "COMMIT", "URL", "VERSION_TAG", "VERSION_TAG_COMMIT", "LLVM_COMMIT", "LLVM_URL",
               "FLATBUFFERS_COMMIT", "FLATBUFFERS_URL", "LLVM_PATCH", "LLVM_PATCH_BLOB", "LLVM_PATCH_SHA256",
               "FLATBUFFERS_CMAKE_FLAGS", "CMAKE", "CMAKE_EXPECT", "NINJA", "NINJA_EXPECT", "CLANG_C", "CLANG_CXX",
               "CLANG_EXPECT", "LINKER", "LINKER_EXPECT", "PYTHON", "PYTHON_EXPECT", "TOOLCHAIN_DIR", "VENV_DIR",
               "TTMLIR_OPT", "TTMLIR_TRANSLATE", "EXPECT_VERSION")
# The bounds of mlir/python/requirements.txt:1-7 at LLVM 4efe170d (lower, upper; None for no bound).
MLIR_BOUNDS = {
    "nanobind": ((2, 9), (3, 0, -1)), "numpy": ((1, 19, 5), (2, 1, 2)), "pybind11": ((2, 10, 0), (2, 13, 6)),
    "pyyaml": ((5, 4, 0), (6, 0, 1)), "ml-dtypes": ((0, 1, 0), (0, 6, 0)), "typing-extensions": ((4, 12, 2), None),
}
# torch 2.7.0+cpu's Requires-Dist for Python 3.10 (the cp310 wheel's metadata, read 2026-10-29).
TORCH_DEPENDENCIES = ("filelock", "typing-extensions", "sympy", "networkx", "jinja2", "fsspec")
TORCH_CP310_SHA256 = "c98c4f48f42a2237e079f3de48e8549de2c8cf68cdcf2041564c7794bbce0b59"
REQUIREMENT = re.compile(
    r"([A-Za-z0-9][A-Za-z0-9._-]*)==([A-Za-z0-9.+!-]+)(?:\s*;[^;]*?)?((?:\s+--hash=sha256:[0-9a-f]{64})+)"
)
FETCHED_ENTRY = re.compile(r"\S+ (?:git [0-9a-f]{40} changed=\d+|files-sha256 [0-9a-f]{64})")
SETTING = re.compile(r"-D([A-Za-z0-9_]+)(?::[A-Z]+)?=(\S+)")
FUNCTION = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\(\) \{\n.*?^\}\n", re.S | re.M)
CORE_LIMIT = "prlimit --pid $$ --core=1:1"
CORE_READ = "resource.getrlimit(resource.RLIMIT_CORE) != (1, 1)"
ALPHA01_ROOT = "/mnt/nvme10/joseph_ufl"
# The values the proposed rx command line gives (plans/spikes/ttmlir-runtime.md), with the gate's layout and HOME
# at the scratch root.
ALPHA01 = {
    "LASSI_ALLOWED_ROOT": ALPHA01_ROOT, "LASSI_STOP_GIB": "345", "LASSI_FREE_FLOOR_GIB": "30",
    "LASSI_TOOLCHAINS": f"{ALPHA01_ROOT}/toolchains", "LASSI_SCRATCH": ALPHA01_ROOT, "LASSI_JOBS": "32",
    "HOME": ALPHA01_ROOT,
}
SCRIPT_VARIABLES = ("LASSI_ALLOWED_ROOT", "LASSI_STOP_GIB", "LASSI_FREE_FLOOR_GIB", "LASSI_TOOLCHAINS",
                    "LASSI_SCRATCH", "LASSI_JOBS")
STAND_INS = ("prlimit", "mkdir", "mktemp", "git", "curl", "cp", "mv", "ln", "rm", "cmake", "ninja", "pip")
SYNTHETIC_COMMIT = "0123456789abcdef0123456789abcdef01234567"


# Helpers.


def pin(name: str = PIN_NAME) -> dict[str, str]:
    """Return a pin file as the repository's pin reader reads it."""
    return read_pin(name)


def script_text() -> str:
    """Return the text of toolchains/tt-mlir-runtime.sh."""
    return (TOOLCHAINS / SCRIPT_NAME).read_text(encoding="utf-8")


def code_lines(text: str) -> list[str]:
    """Return the stripped code lines of bash text, continuations joined, without comments or blank lines."""
    joined = re.sub(r"\\\n\s*", " ", text)
    lines = [line.strip() for line in joined.splitlines()]
    return [line for line in lines if line and not line.startswith("#")]


def logic_lines(text: str) -> list[str]:
    """Return the code lines of bash text that are not echo lines."""
    return [line for line in code_lines(text) if not line.startswith("echo ")]


def functions(text: str) -> dict[str, str]:
    """Return every top-level bash function of a script, its name mapped to its definition through its `}` line."""
    return {match.group(1): match.group(0) for match in FUNCTION.finditer(text)}


def bash_function(text: str, name: str) -> str:
    """Return the definition of the top-level bash function `name`; fail when the script has none."""
    found = functions(text)
    assert name in found, f"the script defines no function {name}"
    return found[name]


def main_section(text: str) -> str:
    """Return the part of a script after its cleanup trap is set: the install steps."""
    return text[text.index("\ntrap cleanup EXIT\n") :]


def calls(line: str, names: set[str]) -> bool:
    """Return whether a code line runs one of the named functions as a command word."""
    return any(re.search(rf"(^|[\s;&|(!]){re.escape(name)}($|[\s;)])", line) for name in names)


def main_steps(text: str) -> list[str]:
    """Return the logic lines of the main section, in order."""
    return logic_lines(main_section(text))


def step_index(steps: list[str], name: str, after: int = -1) -> int:
    """Return the index of the first main step past `after` that runs `name`; fail when none does."""
    for index, line in enumerate(steps):
        if index > after and calls(line, {name}):
            return index
    raise AssertionError(f"the main steps never run {name} after step {after}")


def cmake_settings(flags: str, generator: bool = True) -> dict[str, str]:
    """Return the -DKEY=VALUE settings of a flags value; fail on anything else but a leading -G Ninja."""
    words = flags.split()
    if generator:
        assert words[:2] == ["-G", "Ninja"], "the Ninja generator comes first"
        words = words[2:]
    settings: dict[str, str] = {}
    for word in words:
        match = SETTING.fullmatch(word)
        assert match, f"not a -DKEY=VALUE setting: {word}"
        assert match.group(1) not in settings, f"{match.group(1)} is set twice"
        settings[match.group(1)] = match.group(2)
    return settings


def requirements(name: str) -> dict[str, tuple[str, set[str]]]:
    """Return a tracked requirements file as normalized name -> (version, sha256 set); fail on any other line."""
    text = (TOOLCHAINS / name).read_bytes().decode("ascii")
    joined = re.sub(r"\\\n\s*", " ", text)
    found: dict[str, tuple[str, set[str]]] = {}
    for line in (raw.strip() for raw in joined.splitlines()):
        if not line or line.startswith("#"):
            continue
        match = REQUIREMENT.fullmatch(line)
        assert match, f"{name}: not a name==version line with sha256 hashes: {line!r}"
        key = re.sub(r"[-_.]+", "-", match.group(1)).lower()
        assert key not in found, f"{name} lists {key} twice"
        found[key] = (match.group(2), set(re.findall(r"sha256:([0-9a-f]{64})", match.group(3))))
    return found


def version_tuple(version: str) -> tuple[int, ...]:
    """Return the numeric release part of a version, local label aside: "2.7.0+cpu" gives (2, 7, 0)."""
    return tuple(int(part) for part in version.split("+")[0].split("."))


def run_probe(body: str, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Write `body` to probe.sh in `cwd` and run it in bash."""
    (cwd / "probe.sh").write_bytes(body.encode("ascii"))
    assert BASH is not None
    return subprocess.run([BASH, "probe.sh"], cwd=cwd, capture_output=True, text=True,
                          env={**os.environ, **(env or {})}, timeout=120)


# The globals the function probes set after the pin's own values.
GLOBALS = """set -euo pipefail
here="$PWD/pins"
source "$here/tt-mlir-runtime.pin"
LASSI_TOOLCHAINS="$PWD/tc"
LASSI_SCRATCH="$PWD"
LASSI_JOBS=2
LASSI_STOP_GIB=345
LASSI_FREE_FLOOR_GIB=30
TMPDIR="$PWD/tmp"
prefix="$LASSI_TOOLCHAINS/$PREFIX_NAME"
compiler_prefix="$LASSI_TOOLCHAINS/$COMPILER_PREFIX_NAME"
record_name=lassi-install.txt
unfinished_name=lassi-install.unfinished
fetched_name=lassi-fetched-sources.txt
TTMLIR_TOOLCHAIN_DIR="$prefix/$TOOLCHAIN_DIR"
TTMLIR_VENV_DIR="$prefix/$VENV_DIR"
floor_kib=$((LASSI_FREE_FLOOR_GIB * 1024 * 1024))
"""
# The script's layout lines under the prefix (its `<part>_src=`, `<part>_build=`, and wheelhouse lines).
LAYOUT = re.compile(r'^[a-z]+_(?:src|build)="\$prefix/[^"\n]+"$|^wheelhouse="\$prefix/[^"\n]+"$', re.M)


def run_with_script(body: str, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Run `body` after GLOBALS, the script's layout lines, and its functions, with the pin's files in pins/."""
    (cwd / "pins").mkdir(exist_ok=True)
    for name in ("tt-mlir-runtime.pin", REQUIREMENTS_NAME, TRACKED_FETCHED_NAME):
        shutil.copyfile(TOOLCHAINS / name, cwd / "pins" / name)
    text = script_text()
    layout = LAYOUT.findall(text)
    assert len(layout) == 9, layout
    body = GLOBALS + "".join(f"{line}\n" for line in layout) + "".join(functions(text).values()) + body
    return run_probe(body, cwd, env)


# The pin file.


def test_pin_names_a_build_variant_beside_the_compiler_install() -> None:
    """BUILD runtime gives PREFIX_NAME <NAME>@<VERSION>-runtime, never the compiler install's prefix."""
    values, compiler = pin(), pin("tt-mlir")
    assert (values["NAME"], values["BUILD"]) == ("tt-mlir", "runtime")
    assert values["PREFIX_NAME"] == f"{values['NAME']}@{recorded_version(values)}"
    assert values["PREFIX_NAME"] == "tt-mlir@0.9.0.dev20260221-runtime"
    assert prefix_pin_name(values["PREFIX_NAME"]) == "tt-mlir"
    assert values["COMPILER_PREFIX_NAME"] == compiler["PREFIX_NAME"] != values["PREFIX_NAME"]


@pytest.mark.parametrize("key", SHARED_KEYS)
def test_pin_shares_the_compiler_pins_sources_tools_and_layout(key: str) -> None:
    """The same tt-mlir, LLVM, patch, flatbuffers, host tools, and tool paths as toolchains/tt-mlir.pin."""
    assert pin()[key] == pin("tt-mlir")[key]


def test_pin_turns_on_the_runtime_the_bindings_and_ttrt() -> None:
    """The runtime, TTMetal and TTNN runtimes, bindings, and ttrt on; distributed, system sfpi, and the rest off."""
    settings = cmake_settings(pin()["CMAKE_FLAGS"])
    for name in ON_OPTIONS:
        assert settings.get(name) == "ON", f"{name} (CMakeLists.txt:{TT_MLIR_OPTIONS[name]}) must be ON"
    for name in OFF_OPTIONS:
        assert settings.get(name) == "OFF", f"{name} (CMakeLists.txt:{TT_MLIR_OPTIONS[name]}) must be OFF"
    assert settings.get("TTMLIR_ENABLE_TOOLS", "ON") == "ON", "ttmlir-opt and ttmlir-translate need the tools"
    assert settings["CMAKE_BUILD_TYPE"] == "Release"
    assert "CMAKE_INSTALL_PREFIX" not in settings, "the script places the build; the pin names no path"


def test_pin_names_only_options_tt_mlir_defines_at_the_pin() -> None:
    """A misspelled option would be ignored by CMake and leave a part in its default state."""
    settings = cmake_settings(pin()["CMAKE_FLAGS"])
    ours = {name for name in settings if name.startswith(("TTMLIR_", "TT_RUNTIME_", "TT_USE_", "CODE_COVERAGE"))}
    assert ours <= set(TT_MLIR_OPTIONS), sorted(ours - set(TT_MLIR_OPTIONS))


def test_pin_builds_llvm_with_mlirs_python_bindings_and_test_libraries() -> None:
    """Upstream's LLVM configuration (env/CMakeLists.txt:56-71): bindings on, tests at LLVM's default, on."""
    ours, theirs = cmake_settings(pin()["LLVM_CMAKE_FLAGS"]), cmake_settings(pin("tt-mlir")["LLVM_CMAKE_FLAGS"])
    assert ours["MLIR_ENABLE_BINDINGS_PYTHON"] == "ON", "tools/CMakeLists.txt:6 builds ttrt only with them"
    assert ours.get("LLVM_INCLUDE_TESTS", "ON") == "ON", "python/CMakeLists.txt:144 links MLIR's test translation"
    assert ours["Python3_FIND_VIRTUALENV"] == "ONLY"
    differ = {key for key in set(ours) | set(theirs) if ours.get(key) != theirs.get(key)}
    assert differ <= {"MLIR_ENABLE_BINDINGS_PYTHON", "LLVM_INCLUDE_TESTS"}, differ
    assert "CMAKE_INSTALL_PREFIX" not in ours, "LLVM is never installed"


def test_pin_names_the_tt_metal_of_the_joint_pin() -> None:
    """tt-mlir's own tt-metal is toolchains/tt-metal.pin's commit, submodules, and sfpi, in tt-mlir's third_party."""
    values, metal = pin(), pin("tt-metal")
    assert (values["METAL_COMMIT"], values["METAL_URL"]) == (metal["COMMIT"], metal["URL"])
    assert values["METAL_SUBMODULES"].split() == metal["SUBMODULES"].split()
    for key in ("SFPI_VERSION", "SFPI_FILENAME", "SFPI_SHA256"):
        assert values[key] == metal[key], key
    assert values["METAL_DIR"] == "src/tt-mlir/third_party/tt-metal/src/tt-metal", "third_party/CMakeLists.txt:105"
    assert values["METAL_BUILD_DIR"] == values["METAL_DIR"] + "/build_Release", "third_party/CMakeLists.txt:16"


def test_pin_reads_back_what_tt_mlir_passes_tt_metal() -> None:
    """third_party/CMakeLists.txt:112-126 for this pin's flags: Python bindings, distributed, no Tracy, own sfpi."""
    settings = cmake_settings(pin()["METAL_CACHE_FLAGS"], generator=False)
    assert settings == {"CMAKE_BUILD_TYPE": "Release", "WITH_PYTHON_BINDINGS": "ON", "ENABLE_TRACY": "OFF",
                        "ENABLE_LIBCXX": "OFF", "ENABLE_DISTRIBUTED": "ON", "TT_USE_SYSTEM_SFPI": "OFF"}
    tt_mlir = cmake_settings(pin()["CMAKE_FLAGS"])
    assert settings["TT_USE_SYSTEM_SFPI"] == tt_mlir["TT_USE_SYSTEM_SFPI"]
    assert settings["ENABLE_TRACY"] == tt_mlir["TT_RUNTIME_ENABLE_PERF_TRACE"]


def test_pin_orders_its_builds() -> None:
    """tt-metal's step targets, then the runtime, its module, and the Python packages, then ttrt; never all."""
    values = pin()
    assert (values["METAL_DOWNLOAD_TARGET"], values["METAL_CONFIGURE_TARGET"], values["METAL_TARGET"]) == (
        "tt-metal-download", "tt-metal-configure", "tt-metal")
    targets = values["TARGETS"].split()
    for name in ("ttmlir-opt", "ttmlir-translate", "TTMLIRRuntime", "_ttmlir_runtime", "TTMLIRPythonModules",
                 "TTRTPythonModules"):
        assert name in targets, name
    assert values["TTRT_TARGET"] == "ttrt" and "ttrt" not in targets
    assert not {"all", "install"} & set(targets)


def test_pin_layout_lies_in_the_prefix() -> None:
    """Every path the pin names for the install is relative to the prefix; the runtime module is for Python 3.10."""
    values = pin()
    for key in ("TOOLCHAIN_DIR", "VENV_DIR", "WHEELHOUSE_DIR", "METAL_DIR", "METAL_BUILD_DIR", "RUNTIME_LIBRARY",
                "RUNTIME_MODULE", "PYTHON_PACKAGES", "TTRT", "TTMLIR_OPT", "TTMLIR_TRANSLATE"):
        path = values[key]
        assert path and not path.startswith("/") and ".." not in path.split("/"), key
    major, minor = values["PYTHON_EXPECT"].split()[1].split(".")[:2]
    assert values["RUNTIME_MODULE"].endswith(f"_ttmlir_runtime.cpython-{major}{minor}-x86_64-linux-gnu.so")
    assert values["WHEELHOUSE_DIR"].startswith(values["TOOLCHAIN_DIR"] + "/")
    assert set(values["METAL_LIBRARIES"].split()) >= {"libtt_metal.so", "_ttnncpp.so", "libdevice.so"}


def test_pin_names_the_python_it_builds_with() -> None:
    """The requirements file, the PyTorch CPU index for torch, and the seeded pip and setuptools."""
    values = pin()
    assert values["PYTHON_REQUIREMENTS"] == REQUIREMENTS_NAME
    assert values["PYTHON_EXTRA_INDEX"] == "https://download.pytorch.org/whl/cpu", "tools/ttrt/requirements.txt:1"
    assert (values["PIP_EXPECT"], values["SETUPTOOLS_EXPECT"]) == ("pip 22.0.2", "59.6.0")


def test_pin_plans_space_a_guard_and_the_gate_niceness() -> None:
    """INSTALL_KIB covers the PROJECTED upper end, 29 GiB; the guard polls often; the niceness is the gate's."""
    values = pin()
    assert values["INSTALL_KIB"].isdigit() and int(values["INSTALL_KIB"]) >= 29 * GIB_KIB
    assert values["GUARD_SECONDS"].isdigit() and 0 < int(values["GUARD_SECONDS"]) <= 60
    gate = json.loads(GATE_CONFIG.read_text(encoding="utf-8"))
    assert int(values["NICE_MIN"]) == gate["nice"]
    # tt_metal/distributed/CMakeLists.txt:37-40 at tt-metal 5280a9cf.
    assert values["MPI_LIBRARY"] == "/opt/openmpi-v5.0.7-ulfm/lib/libmpi.so.40"


@needs_bash
def test_bash_reads_the_pin_as_read_pin_does(tmp_path: Path) -> None:
    """The shell that sources the pin sees the values read_pin gives."""
    values = pin()
    shutil.copy(TOOLCHAINS / "tt-mlir-runtime.pin", tmp_path / "tt-mlir-runtime.pin")
    body = "set -euo pipefail\nsource ./tt-mlir-runtime.pin\n" + "".join(f'printf "%s\\n" "${key}"\n' for key in values)
    result = run_probe(body, tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == list(values.values())


# The tracked lists.


def test_requirements_pin_every_package_with_a_sha256() -> None:
    """Every download is pinned by version and checked by sha256; pip and setuptools stay as the venv seeds them."""
    found = requirements(REQUIREMENTS_NAME)
    assert found and all(hashes for _, hashes in found.values())
    assert not {"pip", "setuptools"} & set(found)


def test_requirements_keep_every_compiler_venv_pin() -> None:
    """Each pin of toolchains/tt-mlir-requirements.txt, at the same version with the same hashes."""
    ours, theirs = requirements(REQUIREMENTS_NAME), requirements(COMPILER_REQUIREMENTS_NAME)
    for name, value in theirs.items():
        assert ours.get(name) == value, name


@pytest.mark.parametrize("name", sorted(MLIR_BOUNDS))
def test_requirements_meet_mlirs_python_bounds(name: str) -> None:
    """mlir/python/requirements.txt:1-7 at LLVM 4efe170d: each package inside its bounds."""
    version = version_tuple(requirements(REQUIREMENTS_NAME)[name][0])
    low, high = MLIR_BOUNDS[name]
    assert version >= low, (name, version)
    assert high is None or version <= high, (name, version)


def test_requirements_hold_torch_for_ttrt_and_its_dependencies() -> None:
    """torch 2.7.0+cpu (tools/ttrt/requirements.txt:1) with its cp310 wheel's sha256, and what it requires."""
    found = requirements(REQUIREMENTS_NAME)
    version, hashes = found["torch"]
    assert version == "2.7.0+cpu" and TORCH_CP310_SHA256 in hashes
    assert not [name for name in TORCH_DEPENDENCIES if name not in found]
    assert version_tuple(found["sympy"][0]) >= (1, 13, 3) and version_tuple(found["typing-extensions"][0]) >= (4, 10)


def test_requirements_hold_a_wheel_with_bdist_wheel_for_old_setuptools() -> None:
    """pip 22.0.2's legacy `setup.py bdist_wheel` needs wheel; 0.46.2 restored it for setuptools under 70.1."""
    assert version_tuple(requirements(REQUIREMENTS_NAME)["wheel"][0]) >= (0, 46, 2)


def test_tracked_fetched_sources_list_is_recorded_or_placeholder() -> None:
    """The builds' fetches are a tracked list: PLACEHOLDER until the first install, then its record."""
    path = TOOLCHAINS / TRACKED_FETCHED_NAME
    assert path.read_bytes().startswith(b"#")
    text = path.read_bytes().decode("ascii")
    entries = [line.rstrip() for line in text.splitlines() if line.strip() and not line.startswith("#")]
    if entries in (["PLACEHOLDER"], ["none"]):
        return
    assert entries and all(FETCHED_ENTRY.fullmatch(line) for line in entries), entries


# The files.


@pytest.mark.parametrize("name", FILES)
def test_files_are_plain_ascii_with_lf(name: str) -> None:
    """Plain ASCII and LF line endings, as every tracked toolchain file."""
    data = (TOOLCHAINS / name).read_bytes()
    assert data.isascii() and b"\r" not in data, name


def test_no_file_matches_a_gate_device_pattern() -> None:
    """The gate refuses command text naming a disabled device class; these files stay clear of every pattern."""
    devices = json.loads(GATE_CONFIG.read_text(encoding="utf-8"))["devices"]
    patterns = [pattern for device in devices.values() for pattern in device["patterns"]]
    for name in FILES:
        text = (TOOLCHAINS / name).read_text(encoding="utf-8")
        assert not [pattern for pattern in patterns if re.search(pattern, text)], name


@needs_bash
def test_script_parses_as_bash() -> None:
    """bash -n accepts the script."""
    assert BASH is not None
    result = subprocess.run([BASH, "-n", f"toolchains/{SCRIPT_NAME}"], capture_output=True, text=True, cwd=REPO)
    assert result.returncode == 0, result.stderr


# The script, read as text.


def test_script_is_strict_sources_its_pin_and_needs_every_value() -> None:
    """The tt-mlir.sh preamble, with the free-space floor required too, and no default for any value."""
    text = script_text()
    assert text.startswith("#!/usr/bin/env bash\n") and "set -euo pipefail" in text
    for name in (*SCRIPT_VARIABLES, "HOME"):
        assert f'"${{{name}:?' in text, name
    sourced = [line for line in logic_lines(text) if re.match(r"(source|\.)\s", line)]
    assert sourced == ['source "$here/tt-mlir-runtime.pin"']
    loop = [line for line in logic_lines(text) if line.startswith("for root in ")]
    assert loop == ['for root in "$LASSI_TOOLCHAINS" "$LASSI_SCRATCH" "$HOME"; do']
    assert 'export TMPDIR="$LASSI_SCRATCH/tmp"' in text
    assert 'prefix="$LASSI_TOOLCHAINS/$PREFIX_NAME"' in text
    assert 'compiler_prefix="$LASSI_TOOLCHAINS/$COMPILER_PREFIX_NAME"' in text


def test_header_says_how_the_script_runs_and_what_it_never_touches() -> None:
    """The header names both hosts' routes, the three command-line values, and the compiler install."""
    head = " ".join(line.lstrip("#").strip() for line in script_text().splitlines()[1:40] if line.startswith("#"))
    for word in ("tools/rx.py", "alpha01", "tools/local/host-profile.sh", "tools/local/longjob.py",
                 "LASSI_ALLOWED_ROOT", "LASSI_STOP_GIB", "LASSI_FREE_FLOOR_GIB", "COMPILER_PREFIX_NAME",
                 "opens a device"):
        assert word in head, word


def core_block(text: str) -> str:
    """Return the script's core-limit block, from its `if ! prlimit` line through its `fi` line."""
    start = text.index(f"\nif ! {CORE_LIMIT}") + 1
    return text[start : text.index("\nfi\n", start) + len("\nfi\n")]


def test_script_holds_the_core_limit_at_one_byte_before_any_tool() -> None:
    """prlimit sets 1:1 and the pinned Python reads it back, after the value checks, before every function."""
    text = script_text()
    assert text.count(CORE_LIMIT) == 1 and CORE_READ in core_block(text)
    start = text.index(core_block(text))
    assert start < min(match.start() for match in FUNCTION.finditer(text))
    assert start < text.index('mkdir -p "$TMPDIR"')
    assert not [line for line in logic_lines(text) if re.match(r"ulimit\b", line)]


@needs_bash
@pytest.mark.parametrize(("set_status", "read_status", "holds"), [(0, 0, True), (1, 0, False), (0, 1, False)])
def test_core_limit_block_refuses_unless_set_and_read_back(
    tmp_path: Path, set_status: int, read_status: int, holds: bool
) -> None:
    """The block exits 2 when prlimit fails or the limit does not read (1, 1)."""
    stubs = f"prlimit() {{ return {set_status}; }}\nread_back() {{ return {read_status}; }}\nPYTHON=read_back\n"
    result = run_probe("set -euo pipefail\n" + stubs + core_block(script_text()) + "echo held\n", tmp_path)
    assert (result.returncode, result.stdout) == ((0, "held\n") if holds else (2, "")), result.stderr


def test_script_clears_the_caller_environment_then_sets_its_own() -> None:
    """No compiler, search-path, Python, CPM, tt-metal, or tt-mlir variable of the caller reaches a build."""
    text = script_text()
    loop = text.index("$(compgen -e)")
    pattern = text[loop : text.index("esac", loop)]
    for glob in ("TTMLIR_*", "TT_MLIR_*", "TT_METAL_*", "CMAKE_*", "CPM_*", "PIP_*"):
        assert glob in pattern, glob
    for line in ('export CMAKE_BUILD_PARALLEL_LEVEL="$LASSI_JOBS"', 'export TTMLIR_VENV_DIR="$prefix/$VENV_DIR"',
                 "GIT_TERMINAL_PROMPT=0", "PIP_CONFIG_FILE=/dev/null"):
        assert loop < text.index(line), line


def test_every_build_runs_at_the_gate_job_count_and_is_guarded() -> None:
    """Each Ninja run names -j "$LASSI_JOBS" under run_guarded; flatbuffers' CMake build runs at LASSI_JOBS too."""
    lines = logic_lines(script_text())
    ninja = [line for line in lines if '"$NINJA" -C' in line]
    assert len(ninja) == 6, ninja
    assert all(line.startswith('run_guarded "$NINJA" -C') and '-j "$LASSI_JOBS"' in line for line in ninja), ninja
    builds = [line for line in lines if "--build" in line]
    assert builds and all('--parallel "$LASSI_JOBS"' in line for line in builds), builds
    assert not [line for line in lines if re.search(r"(^|[\s;&|(])(cmake|ninja)\s", line)], "tools by path only"


def test_script_installs_python_packages_only_with_hashes() -> None:
    """pip downloads and installs only the tracked requirements, hash-checked, and installs from the wheelhouse."""
    lines = logic_lines(script_text())
    pips = [line for line in lines if re.search(r"-m pip (download|install)\b", line)]
    assert len(pips) == 2, pips
    for line in pips:
        assert "--require-hashes" in line and '-r "$here/$PYTHON_REQUIREMENTS"' in line, line
        assert "--only-binary :all:" in line and "--no-deps" in line, line
    assert '--extra-index-url "$PYTHON_EXTRA_INDEX"' in pips[0] and "--no-index" in pips[1]


def test_pip_runs_offline_from_the_wheelhouse_after_the_venv() -> None:
    """PIP_NO_INDEX and PIP_FIND_LINKS are set once the venv is made and before the first build."""
    steps = main_steps(script_text())
    offline = steps.index('export PIP_NO_INDEX=1 PIP_FIND_LINKS="$wheelhouse"')
    assert step_index(steps, "make_venv") < offline < step_index(steps, "build_flatbuffers")


def test_script_never_runs_the_environment_build_activate_or_sudo() -> None:
    """No env/CMakeLists.txt, env/activate, venv script, debugger installer, or sudo."""
    lines = logic_lines(script_text())
    for word in ("env/activate", "env/CMakeLists", "init_venv", "install-tt-triage", "install_debugger", "sudo"):
        assert not [line for line in lines if word in line], word


def test_script_writes_no_system_path_and_deletes_only_its_marker() -> None:
    """Only the root-filesystem check names /tmp, /var/tmp, or /opt; the only removal is the start marker's."""
    lines = logic_lines(script_text().partition("\n")[2])
    for system_path in ("/usr/", "/opt/", "/etc/"):
        assert not [line for line in lines if system_path in line], system_path
    naming = [line for line in lines if re.search(r'(^|[\s"=(])/((var/)?tmp|opt)\b', line)]
    assert naming == ['check_root_fs "$marker" /tmp /var/tmp /opt']
    removals = [line for line in lines if re.search(r"(^|[\s;&|(])rm\s", line)]
    assert removals == ['if [ -n "$marker" ]; then rm -f "$marker"; fi']


def test_root_fs_check_is_the_one_tt_metal_sh_uses() -> None:
    """The P4.2 method, read the same way, messages aside."""
    ours = [line for line in logic_lines(bash_function(script_text(), "check_root_fs")) if not line.startswith('"')]
    theirs = logic_lines(bash_function((TOOLCHAINS / "tt-metal.sh").read_text(encoding="utf-8"), "check_root_fs"))
    assert ours == theirs


def test_only_the_guards_name_the_compiler_install() -> None:
    """The compiler prefix is assigned once and read only by the refusal and the untouched check."""
    text = script_text()
    holders = {name for name, body in functions(text).items() if "$compiler_prefix" in body}
    assert holders == {"refuse_compiler_prefix", "check_compiler_untouched"}, holders
    outside = [line for line in logic_lines(main_section(text)) if "$compiler_prefix" in line]
    assert outside == [], outside
    assigned = [line for line in logic_lines(text) if line.startswith("compiler_prefix=")]
    assert assigned == ['compiler_prefix="$LASSI_TOOLCHAINS/$COMPILER_PREFIX_NAME"'], assigned


def test_main_steps_refuse_before_any_download_and_record_last() -> None:
    """The order: the compiler-prefix refusal, the kept install, every refusal, fetches, builds, checks, the record."""
    steps = main_steps(script_text())
    assert steps[0] == "trap cleanup EXIT"
    refuse = step_index(steps, "refuse_compiler_prefix")
    kept = steps.index("exit 0")
    assert refuse == 1 and refuse < kept < next(i for i, line in enumerate(steps) if "mktemp" in line)
    assert any("verify_install" in line for line in steps[refuse:kept]), "a kept install is verified first"
    checks = [step_index(steps, name, kept) for name in ("check_prefix", "check_nice", "check_host", "check_space")]
    first_fetch = step_index(steps, "fetch_at", kept)
    assert checks == sorted(checks) and checks[-1] < first_fetch
    order = ["make_venv", "build_flatbuffers", "build_llvm", "configure_ttmlir", "build_metal", "build_ttmlir",
             "check_tools", "check_runtime", "check_python", "verify_sources", "check_compiler_untouched",
             "check_root_fs"]
    positions = [step_index(steps, name, kept) for name in order]
    assert positions == sorted(positions), dict(zip(order, positions, strict=True))
    record = steps.index('mv -T "$prefix/$unfinished_name" "$prefix/$record_name"')
    assert record > positions[-1]


def test_build_metal_checks_each_step_before_the_next() -> None:
    """The clone is checked, then the configure's cache and sfpi, then the sources are recorded, then the build."""
    body = logic_lines(bash_function(script_text(), "build_metal"))[1:]
    names = ["METAL_DOWNLOAD_TARGET", "verify_metal_tree", "METAL_CONFIGURE_TARGET", "check_metal_cache",
             "check_sfpi", "record_and_compare_sources", '"$METAL_TARGET"', "check_jit_files"]
    found = [next(i for i, line in enumerate(body) if name in line) for name in names]
    assert found == sorted(found), dict(zip(names, found, strict=True))


def test_build_ttmlir_builds_ttrt_after_its_inputs() -> None:
    """TARGETS in one Ninja run, then TTRT_TARGET alone, since ttrt's command copies their outputs."""
    body = logic_lines(bash_function(script_text(), "build_ttmlir"))[1:]
    runs = [line for line in body if line.startswith("run_guarded")]
    assert len(runs) == 2 and '"${targets[@]}"' in runs[0] and '"$TTRT_TARGET"' in runs[1], runs


def test_kept_install_is_verified_in_full() -> None:
    """A kept install passes every check a new one does, the fetched-sources comparison included."""
    body = " ".join(logic_lines(bash_function(script_text(), "verify_install")))
    for name in ("verify_sources", "check_venv", "check_flatc", "check_flatbuffers_cache", "check_llvm_cache",
                 "check_ttmlir_cache", "check_metal_cache", "check_fetched_sources", "check_tools", "check_runtime",
                 "check_python"):
        assert name in body, name


def test_checks_open_no_device_and_import_no_runtime() -> None:
    """ttrt is read through its metadata and ldd, never imported; nothing runs ttrt, a ttsim, or a device tool."""
    lines = logic_lines(script_text())
    runs_ttrt = re.compile(r'(^|[;&|]\s*)"?\$prefix/\$TTRT"?\s|import ttrt|ttrt\.runtime|TT_METAL_SIMULATOR')
    assert not [line for line in lines if runs_ttrt.search(line)]
    assert "importlib.metadata import version" in script_text()


# Behavior: whole-script probes.


def run_script(cwd: Path, values: dict[str, str]) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    """Run the script from `cwd` with only `values` of SCRIPT_VARIABLES (and HOME) set, under logging stand-ins."""
    lines = ["set -euo pipefail", ": >calls.log", 'export CALLS_LOG="$PWD/calls.log"']
    for tool in STAND_INS:
        lines += [f'{tool}() {{ echo "{tool} $*" >>"$CALLS_LOG"; return 97; }}', f"export -f {tool}"]
    lines.append("unset " + " ".join(SCRIPT_VARIABLES))
    lines += [f'export {key}="{value}"' for key, value in values.items()]
    lines.append(f"exec \"$BASH\" '{(TOOLCHAINS / SCRIPT_NAME).as_posix()}'")
    result = run_probe("\n".join(lines) + "\n", cwd)
    return result, (cwd / "calls.log").read_text(encoding="ascii").splitlines()


@needs_bash
def test_script_takes_alpha01s_values_and_sets_the_core_limit_first(tmp_path: Path) -> None:
    """With the proposed command line's values nothing is refused, and prlimit is the first tool to run."""
    result, called = run_script(tmp_path, ALPHA01)
    assert "refusing" not in result.stderr and result.stdout == "", result.stderr
    assert called and called[0].startswith("prlimit ") and "--core=1:1" in called[0], called


@needs_bash
@pytest.mark.parametrize(
    "changed",
    [{"LASSI_TOOLCHAINS": "/mnt/nvme10/other/toolchains"}, {"HOME": "/home/someone"},
     {"LASSI_ALLOWED_ROOT": "relative"}, {"LASSI_FREE_FLOOR_GIB": "0"}, {"LASSI_FREE_FLOOR_GIB": "30G"},
     {"LASSI_FREE_FLOOR_GIB": "030"}, {"LASSI_FREE_FLOOR_GIB": ""}, {"LASSI_STOP_GIB": "-1"}, {"LASSI_JOBS": "0"}],
    ids=["toolchains-outside", "home-outside", "relative-root", "floor-zero", "floor-unit", "floor-octal",
         "floor-empty", "stop-negative", "no-jobs"],
)
def test_script_refuses_bad_values_before_any_tool(tmp_path: Path, changed: dict[str, str]) -> None:
    """Every value is checked before the core limit, so a refusal runs nothing and prints nothing to stdout."""
    result, called = run_script(tmp_path, {**ALPHA01, **changed})
    assert result.returncode != 0 and called == [] and result.stdout == "", (result.stderr, called)
    assert "tt-mlir-runtime" in result.stderr, result.stderr


@needs_bash
def test_script_refuses_without_the_free_space_floor(tmp_path: Path) -> None:
    """LASSI_FREE_FLOOR_GIB has no default: unset, the script stops at once and says where it comes from."""
    values = {key: value for key, value in ALPHA01.items() if key != "LASSI_FREE_FLOOR_GIB"}
    result, called = run_script(tmp_path, values)
    assert result.returncode != 0 and called == [], result.stderr
    assert "LASSI_FREE_FLOOR_GIB" in result.stderr and "command line" in result.stderr


# Behavior: single functions.


SPACE_STUBS = """du() { if [ -n "$DU_KIB" ]; then printf '%s\\t%s\\n' "$DU_KIB" "${@: -1}"; fi; }
df() { printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\\n/dev/x 1 1 %s 1%% /mnt\\n' "$DF_KIB"; }
"""
STOP_KIB = 345 * GIB_KIB
FLOOR_KIB = 30 * GIB_KIB


@needs_bash
@pytest.mark.parametrize(
    ("used", "free", "fits"),
    [(STOP_KIB - 1000, FLOOR_KIB + 1000, True), (STOP_KIB - 999, 10**12, False),
     (1000, FLOOR_KIB + 999, False), ("", 10**12, False)],
    ids=["at-both-lines", "over-the-stop-line", "under-the-floor", "du-fails"],
)
def test_check_space_keeps_the_stop_line_and_the_floor(tmp_path: Path, used: int | str, free: int, fits: bool) -> None:
    """du plus the planned KiB may reach the stop line, and the free space less the planned KiB may reach the floor."""
    result = run_with_script(SPACE_STUBS + 'check_space "1000"\n', tmp_path,
                             env={"DU_KIB": str(used), "DF_KIB": str(free)})
    assert (result.returncode == 0) == fits, result.stdout + result.stderr
    if not fits:
        assert "tt-mlir-runtime:" in result.stderr


GUARD_STUBS = SPACE_STUBS + "GUARD_SECONDS=1\n"


@needs_bash
@pytest.mark.parametrize(
    ("free", "command", "status"),
    [(10**12, "bash -c 'exit 0'", 0), (10**12, "bash -c 'exit 3'", 3), (1000, "sleep 30", 1)],
    ids=["passes", "keeps-the-status", "stops-under-the-floor"],
)
def test_run_guarded_stops_a_build_under_the_floor(tmp_path: Path, free: int, command: str, status: int) -> None:
    """run_guarded returns the command's status, or stops it and fails once the free space is under the floor."""
    body = GUARD_STUBS + f"run_guarded {command} && echo rc=0 || echo rc=$?\n"
    result = run_with_script(body, tmp_path, env={"DF_KIB": str(free)})
    assert result.returncode == 0 and f"rc={status}" in result.stdout, result.stdout + result.stderr
    assert "error" not in result.stderr.replace("tt-mlir-runtime:", ""), result.stderr
    if status == 1:
        assert "under the floor" in result.stderr and "LASSI_FREE_FLOOR_GIB" in result.stderr


@needs_bash
@pytest.mark.parametrize(("case", "kept"), [("apart", True), ("same", False), ("inside", False), ("holds", False)])
def test_refuse_compiler_prefix_keeps_the_two_installs_apart(tmp_path: Path, case: str, kept: bool) -> None:
    """The runtime prefix may neither be, lie in, nor hold the compiler install's prefix."""
    setup = {
        "apart": "",
        "same": 'COMPILER_PREFIX_NAME="$PREFIX_NAME"\ncompiler_prefix="$prefix"\n',
        "inside": 'prefix="$compiler_prefix/runtime"\n',
        "holds": 'compiler_prefix="$prefix/compiler"\n',
    }[case]
    result = run_with_script(setup + "refuse_compiler_prefix\n", tmp_path)
    assert (result.returncode == 0) == kept, result.stdout + result.stderr
    assert not (tmp_path / "tc").exists(), "the check writes nothing"


@needs_bash
@pytest.mark.parametrize(("case", "passes"), [("missing", True), ("older", True), ("newer", False)])
def test_check_compiler_untouched_refuses_a_change_during_the_run(tmp_path: Path, case: str, passes: bool) -> None:
    """A file under the compiler install newer than the run's start marker stops the record."""
    setup = {
        "missing": "touch marker\n",
        "older": 'mkdir -p "$compiler_prefix"\necho SYNTHETIC >"$compiler_prefix/a.txt"\n'
                 'touch -d "2000-01-01" "$compiler_prefix/a.txt" "$compiler_prefix"\ntouch marker\n',
        "newer": 'mkdir -p "$compiler_prefix"\ntouch -d "2000-01-01" marker\n'
                 'echo SYNTHETIC >"$compiler_prefix/a.txt"\n',
    }[case]
    result = run_with_script(setup + "check_compiler_untouched marker\n", tmp_path)
    assert (result.returncode == 0) == passes, result.stdout + result.stderr


@needs_bash
@pytest.mark.parametrize(("level", "passes"), [(0, False), (9, False), (10, True), (19, True)])
def test_check_nice_needs_the_gate_niceness(tmp_path: Path, level: int, passes: bool) -> None:
    """The script runs at NICE_MIN or more, as the gate and the long-job runner run jobs."""
    result = run_with_script(f"nice() {{ echo {level}; }}\ncheck_nice\n", tmp_path)
    assert (result.returncode == 0) == passes, result.stdout + result.stderr


@needs_bash
def test_pinned_record_holds_every_pinned_value_and_writes_nothing(tmp_path: Path) -> None:
    """The record changes with any pinned commit, flag, target, tool, layout value, or Python pin."""
    result = run_with_script("pinned_record >rec1.txt\npinned_record >rec2.txt\n", tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    record = (tmp_path / "rec1.txt").read_text(encoding="ascii")
    assert record == (tmp_path / "rec2.txt").read_text(encoding="ascii")
    values = pin()
    assert record.splitlines()[0].split()[:2] == ["tt-mlir", values["COMMIT"]]
    for key in ("BUILD", "LLVM_COMMIT", "LLVM_PATCH_SHA256", "FLATBUFFERS_COMMIT", "METAL_COMMIT", "METAL_SUBMODULES",
                "SFPI_SHA256", "PYTHON_EXTRA_INDEX", "PIP_EXPECT", "CMAKE_FLAGS", "LLVM_CMAKE_FLAGS",
                "FLATBUFFERS_CMAKE_FLAGS", "METAL_CACHE_FLAGS", "TARGETS", "TTRT_TARGET", "MPI_LIBRARY",
                "RUNTIME_LIBRARY", "WHEELHOUSE_DIR", "METAL_DIR"):
        assert values[key] in record, key
    assert hashlib.sha256((TOOLCHAINS / REQUIREMENTS_NAME).read_bytes()).hexdigest() in record
    assert not (tmp_path / "tc").exists(), "pinned_record writes nothing"


def prefix_state(prefix: Path) -> dict[str, bytes]:
    """Return every file under `prefix` by relative path, with its bytes ({} when the prefix does not exist)."""
    if not prefix.exists():
        return {}
    return {path.relative_to(prefix).as_posix(): path.read_bytes() for path in prefix.rglob("*") if path.is_file()}


@needs_bash
@pytest.mark.parametrize(("case", "kept"), [("absent", True), ("resume", True), ("other-commit", False),
                                            ("unknown", False)])
def test_check_prefix_keeps_only_a_fresh_or_resumable_prefix(tmp_path: Path, case: str, kept: bool) -> None:
    """An unfinished tree at the pin resumes; any other install or content is refused and left as it is."""
    made = run_with_script("pinned_record >rec.txt\n", tmp_path)
    assert made.returncode == 0, made.stderr
    record = (tmp_path / "rec.txt").read_text(encoding="ascii")
    prefix = tmp_path / "tc" / pin()["PREFIX_NAME"]
    contents = {"resume": {UNFINISHED_NAME: record},
                "other-commit": {RECORD_NAME: record.replace(pin()["COMMIT"], SYNTHETIC_COMMIT)},
                "unknown": {"stray.txt": "SYNTHETIC stray file\n"}}.get(case, {})
    for name, content in contents.items():
        prefix.mkdir(parents=True, exist_ok=True)
        (prefix / name).write_bytes(content.encode("ascii"))
    before = prefix_state(prefix)
    result = run_with_script("check_prefix\n", tmp_path)
    assert (result.returncode == 0) == kept, result.stdout + result.stderr
    assert prefix_state(prefix) == before


@needs_bash
@pytest.mark.parametrize(("output", "passes"), [("libtt_metal.so => /x/libtt_metal.so (0x1)\n", True),
                                                ("libtt_metal.so => not found\n", False)],
                         ids=["resolved", "not-found"])
def test_check_ldd_refuses_an_unresolved_library(tmp_path: Path, output: str, passes: bool) -> None:
    """ldd must find every library the runtime and its Python module need."""
    (tmp_path / "lib.so").write_bytes(b"SYNTHETIC\n")
    result = run_with_script(f"ldd() {{ printf '{output}'; }}\ncheck_ldd lib.so\n", tmp_path)
    assert (result.returncode == 0) == passes, result.stdout + result.stderr


PYTHON_STUB = """#!/usr/bin/env bash
case "$3" in
  *ttmlir*) echo "ttmlir.ir from SYNTHETIC" ;;
  *importlib*) echo "$TTRT_READS" ;;
  *sysconfig*) echo "$SITE" ;;
esac
"""


@needs_bash
@pytest.mark.parametrize(("reads", "passes"), [("ttrt 0.0.2995", True), ("ttrt 0.0.2994", False)])
def test_check_python_expects_ttrts_version_from_the_clone(tmp_path: Path, reads: str, passes: bool) -> None:
    """ttrt's version is <VERSION_TAG without v>.<commits since it> (TTMLIRVersion.cmake:44-63; setup.py:11-15)."""
    venv = tmp_path / "tc" / pin()["PREFIX_NAME"] / pin()["VENV_DIR"] / "bin"
    venv.mkdir(parents=True)
    (venv / "python").write_bytes(PYTHON_STUB.encode("ascii"))
    (venv / "python").chmod(0o755)
    (venv / "ttrt").write_bytes(b"#!/usr/bin/env bash\n")
    (venv / "ttrt").chmod(0o755)
    module = tmp_path / "site" / "ttrt" / "runtime" / pin()["RUNTIME_MODULE"].rsplit("/", 1)[1]
    module.parent.mkdir(parents=True)
    module.write_bytes(b"SYNTHETIC\n")
    body = ('git_in() { echo 2995; }\nldd() { echo "libx.so => /x/libx.so"; }\n'
            'export TTRT_READS SITE="$PWD/site"\ncheck_python\n')
    result = run_with_script(body, tmp_path, env={"TTRT_READS": reads})
    assert (result.returncode == 0) == passes, result.stdout + result.stderr


ENTRY_A = "alpha/0001 git " + "1" * 40 + " changed=0"
ENTRY_B = "beta/0002 files-sha256 " + "2" * 64


@needs_bash
@pytest.mark.parametrize(
    ("tree", "tracked", "passes"),
    [(f"{ENTRY_A}\n{ENTRY_B}  \n", f"# SYNTHETIC\n{ENTRY_A}\n{ENTRY_B}\n", True),
     (f"{ENTRY_A}\n", f"# SYNTHETIC\n{ENTRY_A}\n{ENTRY_B}\n", False),
     (f"{ENTRY_A}\n", "# SYNTHETIC\nPLACEHOLDER\n", False)],
    ids=["same", "missing-line", "placeholder"],
)
def test_check_fetched_sources_compares_line_for_line(tmp_path: Path, tree: str, tracked: str, passes: bool) -> None:
    """The tree's list equals the tracked one, comments, blanks, and trailing spaces aside; PLACEHOLDER refuses."""
    (tmp_path / "tree.txt").write_bytes(tree.encode("ascii"))
    (tmp_path / "tracked.txt").write_bytes(tracked.encode("ascii"))
    result = run_with_script("check_fetched_sources tree.txt tracked.txt\n", tmp_path)
    assert (result.returncode == 0) == passes, result.stdout + result.stderr


@needs_bash
@pytest.mark.parametrize(
    ("drop", "key"),
    [(None, None), ("ENABLE_DISTRIBUTED", "ENABLE_DISTRIBUTED"), ("TT_USE_SYSTEM_SFPI", "TT_USE_SYSTEM_SFPI")],
    ids=["as-passed", "no-distributed", "system-sfpi"],
)
def test_check_cache_reads_tt_metals_settings_back(tmp_path: Path, drop: str | None, key: str | None) -> None:
    """tt-metal's cache must hold what tt-mlir passes it for this pin (third_party/CMakeLists.txt:112-126)."""
    settings = cmake_settings(pin()["METAL_CACHE_FLAGS"], generator=False)
    lines = ["# SYNTHETIC CMake cache"]
    for name, value in settings.items():
        if name == drop:
            value = "ON" if value == "OFF" else "OFF"
        lines.append(f"{name}:BOOL={value}" if name != "CMAKE_BUILD_TYPE" else f"{name}:STRING={value}")
    (tmp_path / "CMakeCache.txt").write_bytes(("\n".join(lines) + "\n").encode("ascii"))
    result = run_with_script('check_cache CMakeCache.txt "$METAL_CACHE_FLAGS"\n', tmp_path)
    assert (result.returncode == 0) == (key is None), result.stdout + result.stderr
    if key is not None:
        assert key in result.stderr
