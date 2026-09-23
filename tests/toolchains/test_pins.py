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

from lassi.toolchains.pins import PINS_DIR, read_pin

REPO = Path(__file__).resolve().parents[2]
TOOLCHAINS = REPO / "toolchains"
NAMES = ("cuda", "nvhpc")
REQUIRED_KEYS = ("NAME", "VERSION", "URL", "INSTALL_FLAGS", "PREFIX_NAME", "EXPECT_VERSION")


def test_pin_parser_reads_the_committed_pin_files() -> None:
    # read_pin (lassi.toolchains.pins, which the stage runner uses) parses the files these tests check.
    assert PINS_DIR.resolve() == TOOLCHAINS.resolve()


def test_pin_parser_refuses_a_bad_line_and_a_name_outside_the_pin_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lassi.toolchains import pins

    (tmp_path / "broken.pin").write_bytes(b"# fixture pin\nNAME=broken\nnot a pin line\n")
    monkeypatch.setattr(pins, "PINS_DIR", tmp_path)
    with pytest.raises(ValueError, match=r"broken\.pin:3"):
        pins.read_pin("broken")
    for name in ("../cuda", "sub/cuda", ""):
        with pytest.raises(ValueError, match="pin name"):
            pins.read_pin(name)


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
    assert re.fullmatch(r"[0-9a-f]{64}", pin["SHA256"])
    assert pin["CUDA_HOME_FROM"] == read_pin("cuda")["PREFIX_NAME"]
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
