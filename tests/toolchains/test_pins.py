"""Tests for the CUDA and NVHPC pin files and install scripts (P0.7).

Pin files live in toolchains/<name>.pin and record the version, the install
flags, and the install path (bible Toolchain Pins). They are shell-sourced
KEY=value files, read by toolchains/<name>.sh, which installs in user space
under $LASSI_TOOLCHAINS/<name>@<version> (AGENTS.md Remote Execution) and is
idempotent. These tests read the files only; the installs run on the build
host through tools/rx.py. No value here is a measurement: EXPECT_VERSION holds
the version string the installed compiler must print, taken from the P0.6
spike (plans/spikes/p0-nvcc.md) for nvcc.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TOOLCHAINS = REPO / "toolchains"
NAMES = ("cuda", "nvhpc")
REQUIRED_KEYS = ("NAME", "VERSION", "URL", "INSTALL_FLAGS", "PREFIX_NAME", "EXPECT_VERSION")


def read_pin(name: str) -> dict[str, str]:
    """Return the KEY=value pairs of toolchains/<name>.pin, with surrounding quotes removed."""
    pairs: dict[str, str] = {}
    for line in (TOOLCHAINS / f"{name}.pin").read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Z][A-Z0-9_]*)=(\"[^\"]*\"|[^\s\"#]*)(\s+#.*)?", stripped)
        assert match, f"{name}.pin: not a KEY=value line: {line!r}"
        pairs[match.group(1)] = match.group(2).strip('"')
    return pairs


def script_text(name: str) -> str:
    """Return the text of toolchains/<name>.sh."""
    return (TOOLCHAINS / f"{name}.sh").read_text(encoding="utf-8")


@pytest.mark.parametrize("name", NAMES)
def test_pin_records_version_flags_and_install_path(name: str) -> None:
    pin = read_pin(name)
    for key in REQUIRED_KEYS:
        assert pin.get(key), f"{name}.pin lacks {key}"
    assert pin["NAME"] == name
    assert pin["PREFIX_NAME"] == f"{name}@{pin['VERSION']}"
    assert pin["URL"].startswith("https://developer.download.nvidia.com/")


def test_cuda_pin_carries_the_published_md5_and_the_spike_version() -> None:
    pin = read_pin("cuda")
    assert pin["VERSION"] == "12.6.3"
    assert re.fullmatch(r"[0-9a-f]{32}", pin["MD5"])
    assert pin["URL"].endswith("/cuda_12.6.3_560.35.05_linux.run")
    assert "V12.6.85" in pin["EXPECT_VERSION"]
    for flag in ("--silent", "--toolkit", "--override"):
        assert flag in pin["INSTALL_FLAGS"].split()


def test_nvhpc_pin_is_a_2024_release_with_a_checksum_slot() -> None:
    pin = read_pin("nvhpc")
    assert pin["VERSION"].startswith("24.")
    assert re.fullmatch(r"[0-9a-f]{64}|PLACEHOLDER", pin["SHA256"])
    assert pin["COMPILER_SUBDIR"] == f"Linux_x86_64/{pin['VERSION']}/compilers/bin"
    assert "NVHPC_SILENT=true" in pin["INSTALL_FLAGS"].split()


@pytest.mark.parametrize("name", NAMES)
def test_script_is_strict_sources_its_pin_and_needs_the_gate_environment(name: str) -> None:
    text = script_text(name)
    assert text.startswith("#!/usr/bin/env bash\n")
    assert "set -euo pipefail" in text
    assert f'"$here/{name}.pin"' in text
    assert '"${LASSI_TOOLCHAINS:?' in text and '"${LASSI_SCRATCH:?' in text
    assert "/mnt/nvme10/*" in text, "the script must refuse an install root off the scratch disk (Agent Rule 7)"
    assert 'for root in "$LASSI_TOOLCHAINS" "$LASSI_SCRATCH"' in text
    assert 'export TMPDIR="$LASSI_SCRATCH/tmp"' in text
    assert 'prefix="$LASSI_TOOLCHAINS/$PREFIX_NAME"' in text


@pytest.mark.parametrize("name", NAMES)
def test_script_skips_a_finished_install_before_downloading(name: str) -> None:
    text = script_text(name)
    skip = text.index("already holds the pinned")
    download = text.index("curl ")
    assert skip < download


@pytest.mark.parametrize("name", NAMES)
def test_script_never_needs_root_or_system_paths(name: str) -> None:
    text = script_text(name)
    assert "sudo" not in text
    for system_path in ("/usr/local", "/opt/", "/etc/"):
        assert system_path not in text


def test_cuda_script_clears_the_installer_log_off_the_root_filesystem_on_every_exit() -> None:
    text = script_text("cuda")
    trap = text.index("trap clear_root_log EXIT")
    assert trap < text.index('sh "$runfile"'), "the trap must be set before the installer runs"
    assert "/tmp/cuda-installer.log" in text and 'mv "$log" "$TMPDIR/cuda-installer.log"' in text
    assert not re.search(r'(^|[\s"=])/tmp\b', script_text("nvhpc")), "nvhpc.sh must not name the absolute /tmp"


@pytest.mark.parametrize("name", NAMES)
def test_script_verifies_the_download_and_the_installed_version(name: str) -> None:
    text = script_text(name)
    assert ("md5sum -c" in text) if name == "cuda" else ("sha256sum" in text)
    assert text.count("$EXPECT_VERSION") >= 2, "checked before skipping and after installing"


@pytest.mark.parametrize("name", NAMES)
def test_files_are_plain_ascii_with_lf(name: str) -> None:
    for path in (TOOLCHAINS / f"{name}.pin", TOOLCHAINS / f"{name}.sh"):
        data = path.read_bytes()
        assert data.isascii(), path
        assert b"\r" not in data, path


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is not available")
@pytest.mark.parametrize("name", NAMES)
def test_script_parses_as_bash(name: str) -> None:
    # A relative POSIX path, since bash on Windows would read backslashes in a native path as escapes.
    result = subprocess.run(["bash", "-n", f"toolchains/{name}.sh"], capture_output=True, text=True, cwd=REPO)
    assert result.returncode == 0, result.stderr
