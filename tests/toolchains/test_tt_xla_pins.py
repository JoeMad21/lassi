"""Tests for the tt-xla pin, its locked requirements, and its install script (task P17.38).

Bible: Toolchain Pins (pin files and install rules; the joint pin's tt-metal and sfpi), Training Module
(Compute), Host Facts (the 150G cap and the 145G stop line), Agent Rules 7 and 10; plans/spikes/tt-training.md,
Finding 1c and Finding 3, step 1 (the design); OQ-101 (a) for the window.

The design: toolchains/tt-xla.sh makes a venv at $LASSI_TOOLCHAINS/tt-xla@0.9.0.dev20260223001148 with CPython
3.11 from python-build-standalone (the build uv installs), fetched by URL and checked by size and sha256, and
installs toolchains/tt-xla-requirements.txt, a lock uv pip compile wrote with a sha256 for every distribution,
with `uv pip install --require-hashes --no-deps`; the wheel pjrt-plugin-tt, torch 2.9.0+cpu, and torch-xla
2.9.0+git061c1e7 are fetched first by URL into the prefix and checked by size and sha256, then installed from
there. It refuses a wheel whose metadata names another tt-metal commit, links the pinned sfpi 7.25.0 of the
tt-metal install at the pin into the wheel's runtime directory (the wheel carries none, and would else take the
host's /opt/tenstorrent/sfpi), and writes an install record. Nothing touches a device: no Python of the venv
imports a framework, and no device variable is set.

These tests read the files, and run single functions of the script in bash on local files. No test downloads,
installs, or opens a device. Hashes and sizes are upstream readings of 2026-10-26 (the spike's Command 4, the
package indexes' sha256 fragments, python-build-standalone's SHA256SUMS); no value here is a measurement.
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
SCRIPT = TOOLCHAINS / "tt-xla.sh"
REQUIREMENTS = TOOLCHAINS / "tt-xla-requirements.txt"
BASH = shutil.which("bash")
needs_bash = pytest.mark.skipif(BASH is None, reason="bash is not available")

WHEEL_VERSION = "0.9.0.dev20260223001148"
WHEEL_SHA256 = "fbbe6241869e9646633604b213903b9795f9dc2f9192f875a569799689b0c4db"
WHEEL_SIZE = "121492577"
TORCH_SHA256 = "add3e93ecc1eeaa6853f6a973ce60ffb3cb14ed2e80f5055e139b09385dce0a7"
TORCH_XLA_SHA256 = "2868985b007b4ea591a86fd3fc0388b4717c221aff68dbe6ef4ef773eb146f74"
PYTHON_SHA256 = "64427febea27864d136db46c8efe968eb6fa5ca2813ce1dca4bb95aec31cb2e4"
SUMMARY = (
    "commit=ff3a67dde32ad3311258674bd3fa260e11a20e98, tt-mlir-commit=c7b8bd7784307cce46577bca1c2d33fdab31fccc, "
    "tt-metal-commit=5280a9cfb00998fd49667a29523d03aee905c129, built-date=2026-02-21, build-type=release"
)
REQUIREMENT = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(?:==(\S+)| @ (\S+))((?:\s+--hash=sha256:[0-9a-f]{64})+)$")


def pin() -> dict[str, str]:
    """Return toolchains/tt-xla.pin as the repository's pin reader reads it."""
    return read_pin("tt-xla")


def requirements() -> dict[str, tuple[str | None, str | None, list[str]]]:
    """Return name -> (version, url, hashes) for every requirement of the lock, continuation lines joined."""
    text = re.sub(r"\\\n\s*", " ", REQUIREMENTS.read_text(encoding="ascii"))
    found: dict[str, tuple[str | None, str | None, list[str]]] = {}
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = REQUIREMENT.match(line.strip())
        assert match, f"not a hashed requirement: {line!r}"
        name = match.group(1).lower()
        assert name not in found, f"{name} is listed twice"
        found[name] = (match.group(2), match.group(3), re.findall(r"[0-9a-f]{64}", match.group(4)))
    return found


def script_text() -> str:
    """Return the text of toolchains/tt-xla.sh."""
    return SCRIPT.read_text(encoding="ascii")


def code_lines(text: str) -> list[str]:
    """Return the stripped code lines of bash text, continuations joined, without comments or blank lines."""
    joined = re.sub(r"\\\n\s*", " ", text)
    return [line.strip() for line in joined.splitlines() if line.strip() and not line.strip().startswith("#")]


def bash_function(name: str) -> str:
    """Return the definition of the script's top-level function `name`."""
    match = re.search(rf"^{name}\(\) \{{\n.*?^\}}\n", script_text(), re.S | re.M)
    assert match, f"tt-xla.sh defines no function {name}"
    return match.group(0)


def run_function(names: tuple[str, ...], call: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    """Run `call` in bash with the pin sourced, `here` set, and the script's functions `names` defined."""
    body = "\n".join(bash_function(name) for name in names)
    program = f'set -euo pipefail\nhere="{TOOLCHAINS.as_posix()}"\nsource "$here/tt-xla.pin"\n{body}\n{call}\n'
    path = tmp_path / "probe.sh"
    path.write_bytes(program.encode("ascii"))
    assert BASH is not None
    return subprocess.run([BASH, str(path)], capture_output=True, text=True, cwd=tmp_path, timeout=60)


# ---------------------------------------------------------------- the pin


def test_the_pin_names_the_wheel_its_commits_and_the_install_path() -> None:
    values = pin()
    assert (values["NAME"], values["VERSION"]) == ("tt-xla", WHEEL_VERSION)
    assert values["PREFIX_NAME"] == f"tt-xla@{WHEEL_VERSION}"
    assert (values["WHEEL_SHA256"], values["WHEEL_SIZE"]) == (WHEEL_SHA256, WHEEL_SIZE)
    assert values["WHEEL_URL"] == ("https://pypi.eng.aws.tenstorrent.com/pjrt-plugin-tt/"
                                   f"pjrt_plugin_tt-{WHEEL_VERSION}-cp311-cp311-linux_x86_64.whl")
    assert values["TT_XLA_COMMIT"] == "ff3a67dde32ad3311258674bd3fa260e11a20e98"
    assert values["TT_MLIR_COMMIT"] == "c7b8bd7784307cce46577bca1c2d33fdab31fccc"
    assert (values["TORCH_SHA256"], values["TORCH_SIZE"]) == (TORCH_SHA256, "184462550")
    assert (values["TORCH_XLA_SHA256"], values["TORCH_XLA_SIZE"]) == (TORCH_XLA_SHA256, "107174326")
    assert (values["PYTHON_VERSION"], values["PYTHON_MINOR"]) == ("3.11.16", "3.11")
    assert (values["PYTHON_SHA256"], values["PYTHON_SIZE"]) == (PYTHON_SHA256, "30778779")
    assert values["PYTHON_URL"].startswith("https://github.com/astral-sh/python-build-standalone/releases/download/")
    assert values["VENV_DIR"] == "venv" and values["PYTHON_REQUIREMENTS"] == "tt-xla-requirements.txt"
    assert int(values["INSTALL_KIB"]) > 0


def test_the_pin_agrees_with_the_tt_metal_pin() -> None:
    values, metal = pin(), read_pin("tt-metal")
    assert values["TT_METAL_COMMIT"] == metal["COMMIT"]
    assert values["METAL_PREFIX_NAME"] == metal["PREFIX_NAME"]
    assert (values["SFPI_VERSION"], values["SFPI_SHA256"]) == (metal["SFPI_VERSION"], metal["SFPI_SHA256"])


def test_the_lock_hashes_every_distribution_and_pins_the_frameworks() -> None:
    values, found = pin(), requirements()
    assert found["pjrt-plugin-tt"] == (None, values["WHEEL_URL"], [WHEEL_SHA256])
    assert found["torch"] == (None, values["TORCH_URL"], [TORCH_SHA256])
    assert found["torch-xla"] == (None, values["TORCH_XLA_URL"], [TORCH_XLA_SHA256])
    for name, version in (("jax", "0.7.1"), ("jaxlib", "0.7.1"), ("transformers", "4.57.1"), ("peft", "0.18.1")):
        assert found[name][0] == version, name
        assert found[name][2], f"{name} has no hash"
    assert all(hashes for _, _, hashes in found.values())


def test_the_files_are_plain_ascii_with_lf() -> None:
    for path in (TOOLCHAINS / "tt-xla.pin", SCRIPT, REQUIREMENTS):
        data = path.read_bytes()
        assert data.isascii() and b"\r" not in data, path


# ---------------------------------------------------------------- the script, read


def test_the_script_takes_its_roots_with_no_default_and_holds_the_core_limit() -> None:
    text = script_text()
    assert text.startswith("#!/usr/bin/env bash\n") and "set -euo pipefail" in text
    assert 'source "$here/tt-xla.pin"' in text
    for name in ("LASSI_TOOLCHAINS", "LASSI_SCRATCH", "LASSI_ALLOWED_ROOT", "LASSI_STOP_GIB", "HOME"):
        assert f'"${{{name}:?' in text, name
    assert "prlimit --pid $$ --core=1:1" in text and "(1, 1)" in text
    assert 'export TMPDIR="$LASSI_SCRATCH/tmp"' in text
    assert 'prefix="$LASSI_TOOLCHAINS/$PREFIX_NAME"' in text
    assert "UV_PYTHON_DOWNLOADS=never" in text and "UV_NO_CONFIG=1" in text


def test_the_script_installs_only_hashed_wheels_and_never_needs_root() -> None:
    lines = code_lines(script_text())
    installs = [line for line in lines if "uv pip install" in line]
    assert installs and all("--require-hashes" in line and "--no-deps" in line for line in installs), installs
    assert all("--only-binary :all:" in line for line in installs)
    joined = "\n".join(lines)
    for word in ("sudo", "apt", "dpkg", "pip3 install", "python -m pip", "/usr/local"):
        assert word not in joined, word
    assert "check_root_fs \"$marker\" /tmp /var/tmp /opt" in joined


def test_the_script_touches_no_device() -> None:
    joined = "\n".join(code_lines(script_text()))
    for word in ("import torch", "import jax", "torch_xla", "PJRT_DEVICE", "TT_VISIBLE_DEVICES", "/dev/tenstorrent",
                 "/dev/hugepages", "tt-smi"):
        assert word not in joined, f"the install names {word!r}"


def test_the_script_keeps_a_finished_install_before_any_download() -> None:
    main = script_text()[script_text().index("trap cleanup EXIT"):]
    calls = [line for line in main.splitlines() if line.startswith("fetch_checked ")]
    assert calls and main.index("already holds the pinned") < main.index(calls[0])


@needs_bash
def test_the_script_parses_as_bash() -> None:
    done = subprocess.run([BASH, "-n", "toolchains/tt-xla.sh"], capture_output=True, text=True, cwd=REPO)
    assert done.returncode == 0, done.stderr


# ---------------------------------------------------------------- the script's functions, run


@needs_bash
def test_pinned_record_names_the_wheel_first(tmp_path: Path) -> None:
    done = run_function(("pinned_record",), "pinned_record", tmp_path)
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines()[0] == f"tt-xla {WHEEL_VERSION} {WHEEL_SHA256}"


@needs_bash
def test_check_download_checks_size_and_sha256(tmp_path: Path) -> None:
    data = b"SYNTHETIC archive\n"
    path = tmp_path / "file.bin"
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    good = run_function(("check_download",), f'check_download "{path.as_posix()}" {len(data)} {digest}', tmp_path)
    assert good.returncode == 0, good.stderr
    for size, sha in ((len(data) + 1, digest), (len(data), "0" * 64)):
        bad = run_function(("check_download",), f'check_download "{path.as_posix()}" {size} {sha}', tmp_path)
        assert bad.returncode != 0 and "file.bin" in bad.stderr


@needs_bash
@pytest.mark.parametrize(
    ("version", "summary", "passes"),
    [
        pytest.param(WHEEL_VERSION, SUMMARY, True, id="pinned"),
        pytest.param(WHEEL_VERSION, SUMMARY.replace("5280a9cfb00998fd49667a29523d03aee905c129", "1" * 40), False,
                     id="other-tt-metal"),
        pytest.param("0.9.0.dev20260224001247", SUMMARY, False, id="other-version"),
    ],
)
def test_check_wheel_metadata_refuses_another_tt_metal_or_version(
    tmp_path: Path, version: str, summary: str, passes: bool
) -> None:
    metadata = tmp_path / "METADATA"
    metadata.write_bytes(f"Name: pjrt-plugin-tt\nVersion: {version}\nSummary: {summary}\n".encode("ascii"))
    done = run_function(("check_wheel_metadata",), f'check_wheel_metadata "{metadata.as_posix()}"', tmp_path)
    assert (done.returncode == 0) == passes, done.stderr
    if not passes:
        assert "tt-metal" in done.stderr or "version" in done.stderr.lower()


@needs_bash
def test_local_requirements_points_the_three_wheels_at_the_prefix(tmp_path: Path) -> None:
    out = tmp_path / "local.txt"
    wheels = tmp_path / "wheels"
    call = f'local_requirements "$here/$PYTHON_REQUIREMENTS" "{out.as_posix()}" "{wheels.as_posix()}"'
    done = run_function(("local_requirements",), call, tmp_path)
    assert done.returncode == 0, done.stderr
    text = out.read_text(encoding="ascii")
    values = pin()
    for key in ("WHEEL_URL", "TORCH_URL", "TORCH_XLA_URL"):
        assert values[key] not in text, key
        local = f"file://{wheels.as_posix()}/{values[key].rsplit('/', 1)[1]}"
        assert local in text, local
    original = REQUIREMENTS.read_text(encoding="ascii")
    assert text.count("--hash=sha256:") == original.count("--hash=sha256:"), "every hash is kept"


@needs_bash
def test_check_sfpi_version_file_takes_the_pinned_sfpi_only(tmp_path: Path) -> None:
    good = tmp_path / "good"
    good.write_bytes(f"sfpi_version='7.25.0'\nsfpi_x86_64_debian_txz_hash='{pin()['SFPI_SHA256']}'\n".encode("ascii"))
    bad = tmp_path / "bad"
    bad.write_bytes(b"sfpi_version='7.48.0'\nsfpi_x86_64_debian_txz_hash='" + b"0" * 64 + b"'\n")
    ok = run_function(("check_sfpi_version_file",), f'check_sfpi_version_file "{good.as_posix()}"', tmp_path)
    assert ok.returncode == 0, ok.stderr
    refused = run_function(("check_sfpi_version_file",), f'check_sfpi_version_file "{bad.as_posix()}"', tmp_path)
    assert refused.returncode != 0 and "7.25.0" in refused.stderr


# ---------------------------------------------------------------- the whole script, with stand-ins

STAND_INS = """
prlimit() { return 0; }
python3() { return 0; }
curl() {
  if [ "${1:-}" = --version ]; then echo "curl SYNTHETIC"; return 0; fi
  echo "CALLED curl $*" >>"$PROBE_LOG"
  return 1
}
uv() {
  if [ "${1:-}" = --version ]; then echo "uv SYNTHETIC"; return 0; fi
  echo "CALLED uv $*" >>"$PROBE_LOG"
  return 1
}
export -f prlimit python3 curl uv
"""


def run_script(tmp_path: Path, **changes: str) -> tuple[subprocess.CompletedProcess[str], str]:
    """Run toolchains/tt-xla.sh with stand-ins for prlimit, python3, curl, and uv under a SYNTHETIC scratch root."""
    assert BASH is not None
    # The root in bash's own form (on Windows, Git Bash's /c/... form, which the script's absolute-path check takes).
    posix = 'p="$1"; if command -v cygpath >/dev/null; then p="$(cygpath -u "$p")"; fi; realpath -m "$p"'
    root = subprocess.run([BASH, "-c", posix, "_", f"{tmp_path.as_posix()}/root"], capture_output=True, text=True,
                          check=True).stdout.strip()
    log = tmp_path / "calls.log"
    log.write_bytes(b"")
    values = {"LASSI_TOOLCHAINS": f"{root}/toolchains", "LASSI_SCRATCH": f"{root}/scratch",
              "LASSI_ALLOWED_ROOT": root, "LASSI_STOP_GIB": "145", "HOME": f"{root}/home",
              "PROBE_LOG": log.as_posix(), **changes}
    exports = "".join(f'export {key}="{value}"\n' for key, value in values.items())
    probe = tmp_path / "run.sh"
    probe.write_bytes((STAND_INS + exports + f'bash "{SCRIPT.as_posix()}"\n').encode("ascii"))
    done = subprocess.run([BASH, str(probe)], capture_output=True, text=True, timeout=120, cwd=tmp_path)
    return done, log.read_text(encoding="ascii")


@needs_bash
def test_the_script_refuses_without_its_roots_before_anything(tmp_path: Path) -> None:
    done, calls = run_script(tmp_path, LASSI_STOP_GIB="")
    assert done.returncode != 0 and "LASSI_STOP_GIB" in done.stderr
    done, calls = run_script(tmp_path, LASSI_ALLOWED_ROOT="relative")
    assert done.returncode == 2 and "LASSI_ALLOWED_ROOT" in done.stderr
    assert calls == "", "nothing was fetched or installed"


@needs_bash
def test_the_script_refuses_a_missing_tt_metal_install_before_any_download(tmp_path: Path) -> None:
    done, calls = run_script(tmp_path)
    assert done.returncode == 1, done.stdout + done.stderr
    assert "toolchains/tt-metal.sh" in done.stderr
    assert calls == "", "nothing was fetched or installed"


@needs_bash
def test_the_script_refuses_unknown_content_at_the_prefix_and_leaves_it(tmp_path: Path) -> None:
    prefix = tmp_path / "root" / "toolchains" / pin()["PREFIX_NAME"]
    prefix.mkdir(parents=True)
    (prefix / "SYNTHETIC-file").write_bytes(b"left by someone else\n")
    done, calls = run_script(tmp_path)
    assert done.returncode == 1 and "left as it is" in done.stderr
    assert (prefix / "SYNTHETIC-file").read_bytes() == b"left by someone else\n"
    assert calls == ""
