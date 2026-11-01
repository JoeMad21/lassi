"""Tests for the LASSI tt-mlir driver: its pin, its source, and its build script (task P17.48).

Bible: Toolchain Pins (pin files and install rules), Execution Backends (the
ttsim row), Agent Rules 6, 7, 9, and 10; plans/spikes/ttmlir-runtime.md ("The
LASSI driver" and the runs of 2026-10-30).

The design: toolchains/tt-mlir-driver.pin names the driver's source
(toolchains/tt-mlir-driver/lassi_ttmlir_run.cpp) and the harness header it
includes, each by sha256, and a VERSION made of both, so another source gets
another prefix and no install is ever built over; it builds with the
runtime's own compiler and linker against toolchains/tt-mlir-runtime.pin's
install. toolchains/tt-mlir-driver.sh checks those values, holds RLIMIT_CORE
at 1 byte, works only under LASSI_ALLOWED_ROOT, removes nothing, and runs
nothing it builds. The source opens one device and closes it once, and sets
no environment of its own: no .preinit_array, no re-exec, no setenv (the
ttmlir-ttsim executor passes HOME; the owner's answer to OQ-115).

No test compiles or runs anything but `bash -n`; nothing here is a
measurement.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from lassi.toolchains.pins import read_pin

REPO = Path(__file__).resolve().parents[2]
TOOLCHAINS = REPO / "toolchains"
SCRIPT = TOOLCHAINS / "tt-mlir-driver.sh"
BASH = shutil.which("bash")
needs_bash = pytest.mark.skipif(BASH is None, reason="bash is not available")
HEX64 = re.compile(r"[0-9a-f]{64}")


@pytest.fixture
def pin() -> dict[str, str]:
    """Return the driver pin's pairs."""
    return read_pin("tt-mlir-driver")


def source_bytes(pin: dict[str, str]) -> tuple[bytes, bytes]:
    """Return the bytes of the pinned source and header."""
    return (TOOLCHAINS / pin["SOURCE"]).read_bytes(), (REPO / pin["HEADER"]).read_bytes()


def test_the_pin_names_its_source_and_header_by_sha256(pin: dict[str, str]) -> None:
    source, header = source_bytes(pin)
    assert pin["SOURCE_SHA256"] == hashlib.sha256(source).hexdigest(), "refresh the pin after editing the source"
    assert pin["HEADER_SHA256"] == hashlib.sha256(header).hexdigest(), "refresh the pin after editing lassi_io.h"


def test_version_and_prefix_name_the_source_and_header_bytes(pin: dict[str, str]) -> None:
    source, header = source_bytes(pin)
    assert pin["VERSION"] == hashlib.sha256(source + header).hexdigest()[:12]
    assert pin["PREFIX_NAME"] == f"{pin['NAME']}@{pin['VERSION']}"


def test_the_pin_builds_with_the_runtimes_own_compiler_and_linker(pin: dict[str, str]) -> None:
    runtime = read_pin(pin["RUNTIME_PIN"])
    assert pin["RUNTIME_PIN"] == "tt-mlir-runtime" and runtime["BUILD"] == "runtime"
    assert Path(pin["CXX"]).name == runtime["CLANG_CXX"] and Path(pin["LINKER"]).name == runtime["LINKER"]
    assert pin["CXX_EXPECT"] == runtime["CLANG_EXPECT"]
    assert "-std=c++20" in pin["CXXFLAGS"].split() and "-DNDEBUG" in pin["CXXFLAGS"].split()
    assert pin["INCLUDE_DIRS"].split()[0] == "src/tt-mlir/runtime/include"


def test_the_recorded_descriptor_is_a_sha256_or_a_placeholder(pin: dict[str, str]) -> None:
    assert pin["SYSTEM_DESC"] == "ttsim.ttsys"
    assert pin["SYSTEM_DESC_SHA256"] == "PLACEHOLDER" or HEX64.fullmatch(pin["SYSTEM_DESC_SHA256"])


def test_the_files_are_plain_ascii(pin: dict[str, str]) -> None:
    for path in (TOOLCHAINS / pin["SOURCE"], SCRIPT, TOOLCHAINS / "tt-mlir-driver.pin"):
        text = path.read_bytes()
        assert text.isascii() and b"\r" not in text, path


def test_the_source_opens_one_device_and_closes_it_in_one_place(pin: dict[str, str]) -> None:
    text = (TOOLCHAINS / pin["SOURCE"]).read_text(encoding="ascii")
    assert text.count("rt::openMeshDevice(") == 1, "one open per process (Agent Rule 9)"
    assert text.count("rt::closeMeshDevice(") == 1
    assert text.count("rt::getCurrentSystemDesc(") == 1, "the describe mode's one open and close"
    for word in ("TT_METAL_SIMULATOR=", "/dev/tenstorrent", "setenv(", ".preinit_array", "execve(", "HOME="):
        assert word not in text, f"the environment chooses the device, never the driver: {word}"


def test_the_source_refuses_the_mock_descriptor_and_other_schemas(pin: dict[str, str]) -> None:
    text = (TOOLCHAINS / pin["SOURCE"]).read_text(encoding="ascii")
    calls = ("checkSchemaHash()", "getSystemDescAsJson()", 'topLevelMember(minified(desc.asJson()), "system_desc")')
    for call in calls:
        assert call in text, call
    assert '"TTM0"' in text and '"TTNN"' in text


def test_the_script_checks_its_values_and_removes_nothing() -> None:
    text = SCRIPT.read_text(encoding="ascii")
    assert "set -euo pipefail" in text and "prlimit --pid $$ --core=1:1" in text
    assert "LASSI_ALLOWED_ROOT" in text and "*PLACEHOLDER*" in text
    for name in ("SOURCE_SHA256", "HEADER_SHA256", "VERSION", "CXX_EXPECT"):
        assert name in text, name
    assert not re.search(r"(^|[\s;&|])rm\s", text), "agents never delete on the build host"
    assert "mv -T" in text and "lassi-install.unfinished" in text, "the record is written last"


@needs_bash
def test_the_script_parses() -> None:
    assert BASH is not None
    done = subprocess.run([BASH, "-n", str(SCRIPT)], capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
