"""Tests for the tt-metal and ttsim pin files and install scripts (task P4.2).

Bible: Toolchain Pins (the joint pin), Execution Backends (the ttsim row),
Agent Rules 7 and 10; plans/spikes/p4-tt-pins.md holds the pinned values.

toolchains/tt-metal.pin and toolchains/ttsim.pin are shell-sourced KEY=value
files read by toolchains/tt-metal.sh and toolchains/ttsim.sh, which install
under $LASSI_TOOLCHAINS/<name>@<version> on the build host (AGENTS.md, Remote
Execution). These tests read the files, and run single functions of the
scripts in bash on local files and throwaway git repositories, with stand-ins
where a test needs one; no test fetches from the network, builds tt-metal, or
runs a program that could open a device. No value here is a measurement: the
pinned commits and hashes come from the P4.1 spike and from GitHub.
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
NAMES = ("tt-metal", "ttsim")
TT_METAL_COMMIT = "5280a9cfb00998fd49667a29523d03aee905c129"
SUBMODULES = {
    "tt_metal/third_party/tt_llk": "f990966829c2831bf2a58d8ce31779acbb476139",
    "tt_metal/third_party/umd": "0450f1be3a21cbf17adb97517038586a6a3af4c0",
    "tt_metal/third_party/tracy": "0aaefbb689b4c60694edc905545fc4709fd13f6a",
    "models/demos/t3000/llama2_70b/reference/llama": "29125b7ad8b5513eeaa4417ed92892bf39c8bd74",
}
SFPI_SHA256 = "6a8883c448df537d9661e239e67721b1ba2a4d0e8ea7a573070d2f6c6d015093"
TTSIM_SHA256 = "7b10aa05a5297c4a28f274e39526bfe6c69373661d1b4d4e47c4257e44b79507"
DESCRIPTOR_SHA256 = "24fd3dfae80435a7d9113e255d6d6af9cdf83f6ae3b1c385e1d48a24795ccf49"
GATE_EXAMPLE = "add_2_integers_in_riscv"
TIER_A = ("loopback", "eltwise_binary", "eltwise_sfpu", "matmul_single_core", "matmul_multi_core")
# The bash on PATH, resolved here: on Windows a bare "bash" given to subprocess can start the WSL launcher in
# System32 instead, which sees neither this environment nor Windows paths.
BASH = shutil.which("bash")
needs_bash = pytest.mark.skipif(BASH is None, reason="bash is not available")
IDENTITY = {
    "GIT_AUTHOR_NAME": "Pin Test",
    "GIT_AUTHOR_EMAIL": "pin-test@example.invalid",
    "GIT_COMMITTER_NAME": "Pin Test",
    "GIT_COMMITTER_EMAIL": "pin-test@example.invalid",
}


def script_text(name: str) -> str:
    """Return the text of toolchains/<name>.sh."""
    return (TOOLCHAINS / f"{name}.sh").read_text(encoding="utf-8")


def code_lines(text: str) -> list[str]:
    """Return the lines of a script that are not blank and not comments, stripped."""
    return [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]


def main_section(text: str) -> str:
    """Return the part of a script after its cleanup trap is set: the install steps, without the functions."""
    return text[text.index("\ntrap cleanup EXIT\n") :]


def in_order(text: str, steps: list[str]) -> bool:
    """Return whether each step occurs in `text` after the one before it."""
    position = 0
    for step in steps:
        position = text.index(step, position) + len(step)
    return True


def sha(data: bytes) -> str:
    """Return the sha256 hex digest of `data`."""
    return hashlib.sha256(data).hexdigest()


# Pin files.


@pytest.mark.parametrize("name", NAMES)
def test_pin_parses_and_names_its_prefix(name: str) -> None:
    pin = read_pin(name)
    for key in ("NAME", "VERSION", "URL", "PREFIX_NAME", "INSTALL_KIB"):
        assert pin.get(key), f"{name}.pin lacks {key}"
    assert pin["NAME"] == name
    assert pin["PREFIX_NAME"] == f"{name}@{pin['VERSION']}"
    assert prefix_pin_name(pin["PREFIX_NAME"]) == name
    assert pin["URL"].startswith("https://github.com/tenstorrent/")
    assert pin["INSTALL_KIB"].isdigit() and int(pin["INSTALL_KIB"]) > 0


def test_tt_metal_pin_holds_the_joint_pin_commit_and_submodules() -> None:
    pin = read_pin("tt-metal")
    assert pin["COMMIT"] == TT_METAL_COMMIT
    assert pin["VERSION"] == TT_METAL_COMMIT[:8]
    assert pin["URL"] == "https://github.com/tenstorrent/tt-metal.git"
    entries = pin["SUBMODULES"].split()
    assert dict(entry.split(":") for entry in entries) == SUBMODULES
    assert len(entries) == len(SUBMODULES), "a submodule is listed twice"
    # The TurboQuant checkout is named so the script can refuse it, and the install is not inside it.
    assert pin["FOREIGN_CHECKOUT"] == "/mnt/nvme10/joseph_ufl/tt-metal"
    assert int(pin["INSTALL_KIB"]) == 8 * 1024 * 1024, "the spike's planning bound of 8 GiB"


def test_tt_metal_pin_records_every_download_with_its_sha256() -> None:
    pin = read_pin("tt-metal")
    assert pin["SFPI_VERSION"] == "7.25.0"
    assert pin["SFPI_FILENAME"] == f"sfpi_{pin['SFPI_VERSION']}_x86_64_debian.txz"
    release = f"https://github.com/tenstorrent/sfpi/releases/download/{pin['SFPI_VERSION']}"
    assert pin["SFPI_URL"] == f"{release}/{pin['SFPI_FILENAME']}"
    assert pin["SFPI_SHA256"] == SFPI_SHA256
    assert pin["CPM_URL"] == "https://github.com/cpm-cmake/CPM.cmake/releases/download/v0.40.2/CPM.cmake"
    assert pin["CPM_SHA256"] == "c8cdc32c03816538ce22781ed72964dc864b2a34a310d3b7104812a5ca2d835d"


def cmake_settings(flags: str) -> dict[str, str]:
    """Return the -DKEY=VALUE settings of a CMAKE_FLAGS value; fail on anything but those and -G Ninja."""
    words = flags.split()
    assert words[:2] == ["-G", "Ninja"], "the Ninja generator comes first"
    settings: dict[str, str] = {}
    for word in words[2:]:
        match = re.fullmatch(r"-D([A-Za-z0-9_]+)=(\S+)", word)
        assert match, f"not a -DKEY=VALUE setting: {word}"
        assert match.group(1) not in settings, f"{match.group(1)} is set twice"
        settings[match.group(1)] = match.group(2)
    return settings


# The settings the pin's own ttsim CI build configured: build-artifact.yaml at the pin runs build_metal.sh with
# --build-type Release, the clang-20 libstdc++ toolchain, --build-metal-tests, --build-ttnn-tests,
# --build-programming-examples, --enable-ccache, --disable-profiler, and --without-distributed, and
# build_metal.sh adds its defaults (Python bindings, unity builds, light-metal trace on; compile commands and the
# fake kernels target off).
CI_SETTINGS = {
    "CMAKE_BUILD_TYPE": "Release",
    "TT_METAL_BUILD_TESTS": "ON",
    "TTNN_BUILD_TESTS": "ON",
    "BUILD_PROGRAMMING_EXAMPLES": "ON",
    "ENABLE_CCACHE": "TRUE",
    "ENABLE_TRACY": "OFF",
    "ENABLE_DISTRIBUTED": "OFF",
    "WITH_PYTHON_BINDINGS": "ON",
    "TT_UNITY_BUILDS": "ON",
    "TT_ENABLE_LIGHT_METAL_TRACE": "ON",
    "CMAKE_EXPORT_COMPILE_COMMANDS": "OFF",
    "ENABLE_FAKE_KERNELS_TARGET": "OFF",
}
# What this build changes, none of which changes a compile flag of the targets it builds (owner, 2026-09-25:
# the least CPU on alpha01): the Python bindings and the tests, which only add TT-NN code and test
# directories, and ccache, a compiler cache. UMD's build-time lint needs no setting: tt-metal forces it off
# (tt_metal/third_party/CMakeLists.txt:4).
CHANGED = {
    "WITH_PYTHON_BINDINGS": "OFF",
    "TT_METAL_BUILD_TESTS": "OFF",
    "TTNN_BUILD_TESTS": "OFF",
    "ENABLE_CCACHE": "FALSE",
}


def test_tt_metal_pin_keeps_the_ci_configuration_but_what_is_built() -> None:
    pin = read_pin("tt-metal")
    settings = cmake_settings(pin["CMAKE_FLAGS"])
    # The pinned sfpi 7.25.0 is fetched into the tree (the option's default, as in CI); the host's /opt sfpi
    # is another version.
    expected = {**CI_SETTINGS, **CHANGED, "TT_USE_SYSTEM_SFPI": "OFF"}
    assert settings == expected
    # The script sets the paths itself, in the tree; the pin never names them, and nothing is installed.
    for key in ("CMAKE_INSTALL_PREFIX", "CPM_SOURCE_CACHE", "CMAKE_TOOLCHAIN_FILE", "CMAKE_MAKE_PROGRAM"):
        assert key not in settings
    assert pin["TOOLCHAIN_FILE"] == "cmake/x86_64-linux-clang-20-libstdcpp-toolchain.cmake"
    assert pin["BUILD_DIR"] == "build_Release"


def test_tt_metal_pin_builds_the_runtime_the_jit_files_and_the_examples_only() -> None:
    # tt_metal (tt_metal/CMakeLists.txt:3) and hw_toolchain (tt_metal/hw/CMakeLists.txt:483); the script adds
    # metal_example_<name> for each pinned example.
    assert read_pin("tt-metal")["TARGETS"].split() == ["tt_metal", "hw_toolchain"]


def test_tt_metal_pin_lists_the_gate_example_and_tier_a() -> None:
    examples = read_pin("tt-metal")["EXAMPLES"].split()
    assert len(examples) == len(set(examples)), "an example is listed twice"
    assert examples[0] == GATE_EXAMPLE
    assert set(TIER_A) <= set(examples)


def test_tt_metal_pin_names_each_host_tool_with_its_version() -> None:
    pin = read_pin("tt-metal")
    for key in ("CMAKE", "NINJA"):
        assert pin[key].startswith("/"), f"{key} is an absolute path"
    assert (pin["CLANG_C"], pin["CLANG_CXX"], pin["LINKER"]) == ("clang-20", "clang++-20", "ld.lld-20")
    for key in ("CMAKE_EXPECT", "NINJA_EXPECT", "CLANG_EXPECT", "LINKER_EXPECT"):
        assert re.search(r"\d+\.\d+\.\d+", pin[key]), f"{key} names a version"
    assert pin["LINKER_EXPECT"] == "LLD 20.1.8", "the lld-20 20.1.8 the spike found"
    # The pin names no Python: with the bindings off build_metal.sh passes none to CMake. tt_metal still runs
    # python3 from PATH to generate code, and the script prints which one (Agent Rule 10).
    assert not any(key.startswith("PYTHON") for key in pin)


def test_ttsim_pin_holds_the_release_library_and_the_descriptor_beside_it() -> None:
    pin = read_pin("ttsim")
    assert pin["VERSION"] == "v1.3.4"
    assert pin["LIBRARY"] == "libttsim_wh.so"
    assert pin["URL"] == f"https://github.com/tenstorrent/ttsim/releases/download/{pin['VERSION']}/{pin['LIBRARY']}"
    assert (pin["SIZE"], pin["SHA256"]) == ("162016", TTSIM_SHA256)
    assert pin["SOC_DESCRIPTOR"] == "soc_descriptor.yaml"
    assert pin["SOC_DESCRIPTOR_FROM"] == "tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml"
    assert pin["SOC_DESCRIPTOR_SHA256"] == DESCRIPTOR_SHA256
    assert pin["TT_METAL_FROM"] == read_pin("tt-metal")["PREFIX_NAME"]


def section(text: str, heading: str) -> str:
    """Return the body of the markdown section `heading` up to the next heading of any level."""
    start = text.index(f"\n{heading}\n")
    end = text.find("\n#", start + len(heading) + 2)
    return text[start : end if end != -1 else len(text)]


def test_pins_match_the_bible_joint_pin_and_ttsim_row() -> None:
    bible = BIBLE.read_text(encoding="utf-8")
    pins = section(bible, "### Toolchain Pins")
    tt_metal, ttsim = read_pin("tt-metal"), read_pin("ttsim")
    for value in (tt_metal["COMMIT"], tt_metal["SFPI_FILENAME"], tt_metal["SFPI_SHA256"]):
        assert value in pins
    for value in (f"ttsim {ttsim['VERSION']}", ttsim["LIBRARY"], f"{ttsim['SIZE']} bytes", ttsim["SHA256"]):
        assert value in pins
    backends = section(bible, "## Execution Backends")
    assert f"TT_METAL_SIMULATOR={ttsim['LIBRARY']}" in backends
    assert f"`{ttsim['SOC_DESCRIPTOR']}` beside it" in backends


@needs_bash
@pytest.mark.parametrize("name", NAMES)
def test_bash_reads_the_pin_as_read_pin_does(tmp_path: Path, name: str) -> None:
    pin = read_pin(name)
    shutil.copy(TOOLCHAINS / f"{name}.pin", tmp_path / f"{name}.pin")
    body = f"set -euo pipefail\nsource ./{name}.pin\n" + "".join(f'printf "%s\\n" "${key}"\n' for key in pin)
    (tmp_path / "probe.sh").write_bytes(body.encode("ascii"))
    assert BASH is not None
    result = subprocess.run([BASH, "probe.sh"], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == list(pin.values())


# Scripts, read as text.


@pytest.mark.parametrize("name", NAMES)
def test_script_is_strict_sources_its_pin_and_needs_the_gate_environment(name: str) -> None:
    text = script_text(name)
    assert text.startswith("#!/usr/bin/env bash\n")
    assert "set -euo pipefail" in text
    assert f'"$here/{name}.pin"' in text
    assert '"${LASSI_TOOLCHAINS:?' in text and '"${LASSI_SCRATCH:?' in text
    assert 'for root in "$LASSI_TOOLCHAINS" "$LASSI_SCRATCH"' in text
    # The roots, with links resolved, must lie in the scratch root (Agent Rule 7).
    assert 'case "$(realpath -m "$root")/" in\n    /mnt/nvme10/joseph_ufl/*) ;;' in text
    # No core file from a crashing tool reaches the root filesystem.
    assert text.index("\nulimit -c 0\n") < text.index(f'source "$here/{name}.pin"')
    assert 'export TMPDIR="$LASSI_SCRATCH/tmp"' in text
    assert 'prefix="$LASSI_TOOLCHAINS/$PREFIX_NAME"' in text


def test_tt_metal_script_builds_with_the_gate_job_count() -> None:
    text = script_text("tt-metal")
    assert '"${LASSI_JOBS:?' in text
    builds = [line for line in code_lines(text) if "--build" in line]
    assert builds and all('--parallel "$LASSI_JOBS"' in line for line in builds), builds


@pytest.mark.parametrize("name", NAMES)
def test_script_never_needs_root_or_system_paths(name: str) -> None:
    # Host tool paths live in the pin, where they are recorded; the scripts name none.
    shebang, _, text = script_text(name).partition("\n")
    assert shebang == "#!/usr/bin/env bash"
    assert "sudo" not in text
    for system_path in ("/usr/", "/opt/", "/etc/"):
        assert system_path not in text


def test_no_file_matches_a_gate_device_pattern() -> None:
    # The gate refuses command text naming a disabled device class; these files stay clear of every pattern.
    devices = json.loads(GATE_CONFIG.read_text(encoding="utf-8"))["devices"]
    patterns = [pattern for device in devices.values() for pattern in device["patterns"]]
    assert patterns
    for name in NAMES:
        for path in (TOOLCHAINS / f"{name}.pin", TOOLCHAINS / f"{name}.sh"):
            text = path.read_text(encoding="utf-8")
            for pattern in patterns:
                assert not re.search(pattern, text), f"{path.name} matches the gate pattern {pattern}"


@pytest.mark.parametrize("name", NAMES)
def test_only_the_root_fs_check_names_the_system_temporary_directories(name: str) -> None:
    lines = [line for line in code_lines(script_text(name)) if not line.startswith("echo ")]
    naming = [line for line in lines if re.search(r'(^|[\s"=(])/(var/)?tmp\b', line)]
    assert naming == ['check_root_fs "$marker" /tmp /var/tmp']


@pytest.mark.parametrize(("name", "work"), [("tt-metal", 'fetch_at "$prefix"'), ("ttsim", 'fetch "$URL"')])
def test_script_keeps_a_finished_install_before_any_download(name: str, work: str) -> None:
    text = script_text(name)
    assert text.index("already holds the pinned") < text.index(work)


def test_tt_metal_script_refuses_the_foreign_checkout_before_writing_anything() -> None:
    text = script_text("tt-metal")
    guard = text.index('\nrefuse_foreign "$prefix" "$LASSI_SCRATCH/tmp"\n')
    assert guard < text.index('\nmkdir -p "$TMPDIR"\n'), "checked before the first directory is made"
    assert guard < text.index("\ntrap cleanup EXIT\n")


def test_tt_metal_script_checks_then_fetches_builds_verifies_and_records_in_order() -> None:
    main = main_section(script_text("tt-metal"))
    assert in_order(
        main,
        [
            "already holds the pinned",
            "check_host\n",
            'check_space "$INSTALL_KIB"',
            'pinned_record >"$prefix/$unfinished_name"',
            'fetch_at "$prefix" "$URL" "$COMMIT"',
            "fetch_submodules\n",
            "verify_tree\n",
            "check_sfpi_source\n",
            "configure\n",
            "check_cache\n",
            "check_sfpi\n",
            "record_cpm_sources\n",
            "build_tree\n",
            "check_jit_files\n",
            "check_examples\n",
            "check_linked\n",
            "verify_tree\n",
            'check_root_fs "$marker" /tmp /var/tmp',
            'mv -T "$prefix/$unfinished_name" "$prefix/$record_name"',
        ],
    )


def test_ttsim_script_stages_verifies_then_swaps() -> None:
    text = script_text("ttsim")
    assert 'staging="$prefix.staging"' in text
    assert in_order(
        main_section(text),
        [
            'check_space "$INSTALL_KIB"',
            'descriptor="$(source_descriptor)"',
            'fetch "$URL" "$SHA256" "$library"',
            'mkdir "$staging"',
            'installed_matches "$staging"',
            'check_root_fs "$marker" /tmp /var/tmp',
            "\nswap_in\n",
            'installed_matches "$prefix"',
        ],
    )


@pytest.mark.parametrize(
    ("name", "targets"),
    [("tt-metal", {'"$marker"'}), ("ttsim", {'"$marker"', '"$part"', '"$staging"'})],
)
def test_script_removes_only_what_the_run_created(name: str, targets: set[str]) -> None:
    removed = [line for line in code_lines(script_text(name)) if re.search(r"\brm\b", line)]
    assert {re.search(r'rm -r?f ("[^"]*")', line).group(1) for line in removed} == targets, removed


def test_tt_metal_script_runs_no_built_program_and_sets_no_simulator() -> None:
    lines = code_lines(script_text("tt-metal"))
    for line in lines:
        if "metal_example_" in line and "programming_examples" in line:
            # Only file tests and readelf, which reads the file's .comment section.
            tested = re.match(r"(if|\|\|) \[ ! -[fx] ", line) or 'out="$(readelf -p .comment "$build/' in line
            assert tested, f"an example is only tested or read, never run: {line}"
    assert not any("TT_METAL_SIMULATOR" in line for line in lines)


@pytest.mark.parametrize("name", NAMES)
def test_files_are_plain_ascii_with_lf(name: str) -> None:
    for path in (TOOLCHAINS / f"{name}.pin", TOOLCHAINS / f"{name}.sh"):
        data = path.read_bytes()
        assert data.isascii(), path
        assert b"\r" not in data, path


@needs_bash
@pytest.mark.parametrize("name", NAMES)
def test_script_parses_as_bash(name: str) -> None:
    # A relative POSIX path, since bash on Windows would read backslashes in a native path as escapes.
    result = subprocess.run(["bash", "-n", f"toolchains/{name}.sh"], capture_output=True, text=True, cwd=REPO)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("name", NAMES)
def test_root_fs_check_is_the_one_cuda_sh_uses(name: str) -> None:
    # The P0.19 check, as cuda.sh has it; only the message prefix and the name of the step it guards differ.
    ours = bash_function(script_text(name), "check_root_fs").replace(f"{name}:", "cuda:")
    ours = ours.replace("nothing is recorded", "nothing is swapped in")
    assert ours == bash_function(script_text("cuda"), "check_root_fs")


def test_ttsim_fetch_is_the_one_cuda_sh_uses() -> None:
    ours = bash_function(script_text("ttsim"), "fetch").replace("ttsim:", "cuda:")
    assert ours == bash_function(script_text("cuda"), "fetch")


# Behavior: single functions of the scripts, run in bash on local files.


def bash_function(text: str, name: str) -> str:
    """Return the definition of the top-level bash function `name`, from its `name() {` line to its `}` line."""
    match = re.search(rf"^{name}\(\) \{{\n.*?^\}}\n", text, re.S | re.M)
    assert match, f"the script defines no function {name}"
    return match.group(0)


def run_functions(
    script: str, names: tuple[str, ...], body: str, cwd: Path, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run `body` in strict bash in `cwd`, after the named functions of toolchains/<script>.sh."""
    text = script_text(script)
    probe = "set -euo pipefail\n" + "".join(bash_function(text, name) for name in names) + body
    (cwd / "probe.sh").write_bytes(probe.encode("ascii"))
    assert BASH is not None
    return subprocess.run(
        [BASH, "probe.sh"], cwd=cwd, capture_output=True, text=True, env={**os.environ, **IDENTITY, **(env or {})}
    )


def fixture_git(cwd: Path, *args: str) -> str:
    """Run git in a throwaway repository with the test identity and unsigned commits; return its stdout."""
    result = subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "-c", "init.defaultBranch=main", "-C", str(cwd), *args],
        capture_output=True, text=True, check=True, env={**os.environ, **IDENTITY},
    )
    return result.stdout.strip()


def make_repo(path: Path, files: list[dict[str, str]]) -> list[str]:
    """Create a repository at `path` with one commit per mapping of file name to text; return the commit ids."""
    path.mkdir(parents=True)
    fixture_git(path, "init", "-q")
    commits = []
    for snapshot in files:
        for name, content in snapshot.items():
            (path / name).parent.mkdir(parents=True, exist_ok=True)
            (path / name).write_bytes(content.encode("ascii"))
            fixture_git(path, "add", name)
        fixture_git(path, "commit", "-q", "-m", f"commit {len(commits)}")
        commits.append(fixture_git(path, "rev-parse", "HEAD"))
    return commits


FETCH = ("git_in", "fetch_at")


# The probes give fetch_at absolute directories, as the script does ($prefix is absolute): git runs with -C
# <dir>, so a URL such as ../up is read relative to <dir>.


@needs_bash
def test_fetch_at_fetches_the_commit_alone_then_keeps_it(tmp_path: Path) -> None:
    commits = make_repo(tmp_path / "up", [{"f": "1\n"}, {"f": "2\n"}, {"f": "3\n"}])
    fetch = f'fetch_at "$prefix" ../up {commits[1]}\n'
    result = run_functions("tt-metal", FETCH, 'prefix="$PWD/down"\n' + fetch + fetch, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert fixture_git(tmp_path / "down", "rev-parse", "HEAD") == commits[1]
    assert fixture_git(tmp_path / "down", "rev-list", "--count", "HEAD") == "1", "fetched at depth 1"
    assert fixture_git(tmp_path / "down", "tag") == "", "no tags are fetched"
    assert f"fetched ../up at {commits[1]}" in result.stdout
    assert f"down is already at {commits[1]}" in result.stdout, "a second run keeps the checkout"


@needs_bash
@pytest.mark.parametrize("case", ["other-commit", "other-origin"])
def test_fetch_at_refuses_a_checkout_it_cannot_keep_and_leaves_it(tmp_path: Path, case: str) -> None:
    commits = make_repo(tmp_path / "up", [{"f": "1\n"}, {"f": "2\n"}])
    first = f'prefix="$PWD/down"\nfetch_at "$prefix" ../up {commits[0]}\n'
    assert run_functions("tt-metal", FETCH, first, tmp_path).returncode == 0
    origin = "../up" if case == "other-commit" else "../elsewhere"
    second = f'prefix="$PWD/down"\nfetch_at "$prefix" {origin} {commits[1]}\n'
    result = run_functions("tt-metal", FETCH, second, tmp_path)
    assert result.returncode != 0
    assert "never delete it" in result.stderr
    assert fixture_git(tmp_path / "down", "rev-parse", "HEAD") == commits[0], "the checkout is left as it was"


SFPI_LINES = f"sfpi_version='7.25.0'\nsfpi_x86_64_debian_txz_hash='{SFPI_SHA256}'\n"


def make_superproject(tmp_path: Path) -> tuple[str, str, str]:
    """Create up/ with tt_metal/sfpi-version and a gitlink sub -> sub-up/ (at its first of two commits).

    The submodule URL is absolute, as tt-metal's are. Returns the superproject
    commit, the submodule's pinned commit, and its later commit.
    """
    sub = make_repo(tmp_path / "sub-up", [{"s": "1\n"}, {"s": "2\n"}])
    gitmodules = f'[submodule "sub"]\n\tpath = sub\n\turl = {(tmp_path / "sub-up").as_posix()}\n'
    up = tmp_path / "up"
    make_repo(up, [{"tt_metal/sfpi-version": SFPI_LINES, ".gitmodules": gitmodules}])
    fixture_git(up, "update-index", "--add", "--cacheinfo", f"160000,{sub[0]},sub")
    fixture_git(up, "commit", "-q", "-m", "add the submodule")
    return fixture_git(up, "rev-parse", "HEAD"), sub[0], sub[1]


def tree_body(commit: str, submodules: str, steps: str) -> str:
    """Return a probe body that sets the pin values the tree functions read, then runs `steps`."""
    return (
        f"prefix=$PWD/down\nCOMMIT={commit}\nSUBMODULES='{submodules}'\n"
        f"SFPI_VERSION=7.25.0\nSFPI_SHA256={SFPI_SHA256}\n"
        f'fetch_at "$prefix" ../up "$COMMIT"\n{steps}'
    )


TREE = ("git_in", "fetch_at", "fetch_submodules", "verify_tree")


@needs_bash
def test_fetch_submodules_fetches_each_pinned_gitlink_and_the_tree_verifies(tmp_path: Path) -> None:
    commit, pinned, _ = make_superproject(tmp_path)
    result = run_functions("tt-metal", TREE, tree_body(commit, f"sub:{pinned}", "fetch_submodules\nverify_tree\n"),
                           tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert fixture_git(tmp_path / "down" / "sub", "rev-parse", "HEAD") == pinned
    assert fixture_git(tmp_path / "down" / "sub", "rev-list", "--count", "HEAD") == "1"
    assert "unchanged" in result.stdout


@needs_bash
def test_fetch_submodules_refuses_gitlinks_other_than_the_pinned_ones(tmp_path: Path) -> None:
    commit, _, later = make_superproject(tmp_path)
    result = run_functions("tt-metal", TREE, tree_body(commit, f"sub:{later}", "fetch_submodules\n"), tmp_path)
    assert result.returncode != 0
    assert "not the pinned SUBMODULES" in result.stderr
    assert not (tmp_path / "down" / "sub" / ".git").exists(), "nothing is fetched for a pin the tree contradicts"


@needs_bash
@pytest.mark.parametrize("change", ["tracked-edit", "sfpi-version"])
def test_verify_tree_refuses_a_changed_tree(tmp_path: Path, change: str) -> None:
    commit, pinned, _ = make_superproject(tmp_path)
    edit = "echo edited >down/tt_metal/sfpi-version\n"
    body = tree_body(commit, f"sub:{pinned}", f"fetch_submodules\n{edit}verify_tree\n")
    if change == "sfpi-version":
        body = body.replace("SFPI_VERSION=7.25.0", "SFPI_VERSION=7.29.0").replace(edit, "")
    result = run_functions("tt-metal", TREE, body, tmp_path)
    assert result.returncode != 0
    assert ("differ from the pinned commits" if change == "tracked-edit" else "sfpi 7.29.0") in result.stderr


def cache_body(flags: str, cache_lines: list[str]) -> str:
    """Return a probe body that writes b/CMakeCache.txt and runs check_cache with the given CMAKE_FLAGS."""
    cache = "\\n".join(cache_lines)
    return (
        "prefix=/t\nbuild=b\ncpm_cache=/t/.cpmcache\nTOOLCHAIN_FILE=cmake/tc.cmake\nNINJA=/n\n"
        f"CMAKE_FLAGS='{flags}'\nmkdir -p b\nprintf '{cache}\\n' >b/CMakeCache.txt\ncheck_cache\n"
    )


PATH_CACHE = [
    "CMAKE_TOOLCHAIN_FILE:UNINITIALIZED=/t/cmake/tc.cmake",
    "CMAKE_MAKE_PROGRAM:FILEPATH=/n",
    "CPM_SOURCE_CACHE:STRING=/t/.cpmcache",
    "CMAKE_FIND_USE_PACKAGE_REGISTRY:UNINITIALIZED=OFF",
    "CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY:UNINITIALIZED=OFF",
    "CMAKE_GENERATOR:INTERNAL=Ninja",
]


@needs_bash
def test_check_cache_passes_when_every_pinned_setting_is_configured(tmp_path: Path) -> None:
    body = cache_body("-G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_TRACY=OFF",
                      ["CMAKE_BUILD_TYPE:STRING=Release", "ENABLE_TRACY:BOOL=OFF", *PATH_CACHE])
    result = run_functions("tt-metal", ("local_settings", "check_cache"), body, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


@needs_bash
@pytest.mark.parametrize(
    ("cache", "key"),
    [
        (["CMAKE_BUILD_TYPE:STRING=RelWithDebInfo", "ENABLE_TRACY:BOOL=OFF", *PATH_CACHE], "CMAKE_BUILD_TYPE"),
        (["CMAKE_BUILD_TYPE:STRING=Release", *PATH_CACHE], "ENABLE_TRACY"),
        (["CMAKE_BUILD_TYPE:STRING=Release", "ENABLE_TRACY:BOOL=OFF", *PATH_CACHE[:2], *PATH_CACHE[3:]],
         "CPM_SOURCE_CACHE"),
    ],
    ids=["other-value", "missing-flag", "missing-path"],
)
def test_check_cache_refuses_a_setting_the_pin_does_not_record(tmp_path: Path, cache: list[str], key: str) -> None:
    body = cache_body("-G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_TRACY=OFF", cache)
    result = run_functions("tt-metal", ("local_settings", "check_cache"), body, tmp_path)
    assert result.returncode != 0
    assert key in result.stderr


def sfpi_body(banner: str, archive: bytes | None) -> str:
    """Return a probe body with a stand-in sfpi compiler printing `banner`, and a kept download if given."""
    body = (
        "prefix=$PWD/t\nbuild=$PWD/t/b\nSFPI_VERSION=7.25.0\nSFPI_FILENAME=sfpi.txz\n"
        f"SFPI_SHA256={sha(b'sfpi archive')}\nmkdir -p t/runtime/sfpi/compiler/bin t/b/_deps/s\n"
        f"printf '#!/bin/sh\\necho \"{banner}\"\\n' >t/runtime/sfpi/compiler/bin/riscv-tt-elf-g++\n"
        "chmod +x t/runtime/sfpi/compiler/bin/riscv-tt-elf-g++\n"
    )
    if archive is not None:
        body += f"printf '{archive.decode('ascii')}' >t/b/_deps/s/sfpi.txz\n"
    return body + "check_sfpi\n"


@needs_bash
@pytest.mark.parametrize(
    ("banner", "archive", "passes"),
    [
        ("riscv-tt-elf-g++ (tenstorrent/sfpi:7.25.0[252]) 15.1.0", None, True),
        ("riscv-tt-elf-g++ (tenstorrent/sfpi:7.25.0[252]) 15.1.0", b"sfpi archive", True),
        ("riscv-tt-elf-g++ (tenstorrent/sfpi:7.25.0[252]) 15.1.0", b"another archive", False),
        ("riscv-tt-elf-g++ (tenstorrent/sfpi:7.29.0[259]) 15.1.0", None, False),
        ("riscv-tt-elf-g++ (tenstorrent/sfpi:7.25.01[1]) 15.1.0", None, False),
    ],
    ids=["version", "version-and-archive", "archive-sha256", "other-version", "longer-version"],
)
def test_check_sfpi_reads_the_version_as_cmake_does(
    tmp_path: Path, banner: str, archive: bytes | None, passes: bool
) -> None:
    result = run_functions("tt-metal", ("check_sfpi",), sfpi_body(banner, archive), tmp_path)
    assert (result.returncode == 0) == passes, result.stdout + result.stderr


@needs_bash
@pytest.mark.parametrize("missing", [None, "loopback"])
def test_check_examples_needs_every_pinned_example_built(tmp_path: Path, missing: str | None) -> None:
    examples = [GATE_EXAMPLE, *TIER_A]
    made = "".join(
        f"printf '#!/bin/sh\\n' >b/programming_examples/metal_example_{name}\n"
        f"chmod +x b/programming_examples/metal_example_{name}\n"
        for name in examples
        if name != missing
    )
    body = f"build=b\nEXAMPLES='{' '.join(examples)}'\nmkdir -p b/programming_examples\n{made}check_examples\n"
    result = run_functions("tt-metal", ("check_examples",), body, tmp_path)
    assert (result.returncode == 0) == (missing is None), result.stdout + result.stderr
    if missing:
        assert f"metal_example_{missing}" in result.stderr


@needs_bash
def test_build_tree_builds_only_the_pinned_targets_in_one_run(tmp_path: Path) -> None:
    pin = read_pin("tt-metal")
    stub = 'stub_cmake() { printf "%s\\n" "$@" >>cmake.log; }\n'
    body = (
        f"{stub}CMAKE=stub_cmake\nbuild=b\nLASSI_JOBS=7\nTARGETS='{pin['TARGETS']}'\n"
        f"EXAMPLES='{pin['EXAMPLES']}'\nbuild_tree\n"
    )
    result = run_functions("tt-metal", ("build_tree",), body, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    examples = [f"metal_example_{name}" for name in pin["EXAMPLES"].split()]
    expected = ["--build", "b", "--target", "tt_metal", "hw_toolchain", *examples, "--parallel", "7"]
    assert (tmp_path / "cmake.log").read_text().split() == expected, "one run, nothing but the pinned targets"


@needs_bash
@pytest.mark.parametrize("missing", [None, "toolchain/wormhole/brisc.ld", "lib/wormhole/crt0.o"])
def test_check_jit_files_needs_the_wormhole_linker_scripts_and_objects(tmp_path: Path, missing: str | None) -> None:
    for name in ("toolchain/wormhole/brisc.ld", "lib/wormhole/crt0.o", "lib/blackhole/crt0.o"):
        if name != missing:
            (tmp_path / "t" / "runtime" / "hw" / name).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / "t" / "runtime" / "hw" / name).write_bytes(b"x")
    result = run_functions("tt-metal", ("check_jit_files",), 'prefix="$PWD/t"\ncheck_jit_files\n', tmp_path)
    assert (result.returncode == 0) == (missing is None), result.stdout + result.stderr


@needs_bash
@pytest.mark.parametrize(
    ("path", "refused"),
    [("foreign", True), ("foreign/toolchains/tt-metal@x", True), ("foreign-other", False), ("toolchains/x", False)],
)
def test_refuse_foreign_keeps_writes_out_of_the_other_checkout(tmp_path: Path, path: str, refused: bool) -> None:
    body = f'FOREIGN_CHECKOUT="$PWD/foreign"\nrefuse_foreign "$PWD/toolchains" "$PWD/{path}"\n'
    result = run_functions("tt-metal", ("refuse_foreign",), body, tmp_path)
    assert (result.returncode != 0) == refused, result.stdout + result.stderr


@needs_bash
def test_move_aside_keeps_the_tree_and_refuses_a_taken_name(tmp_path: Path) -> None:
    (tmp_path / "tc" / "tt-metal@x").mkdir(parents=True)
    (tmp_path / "tc" / "tt-metal@x" / "kept.txt").write_bytes(b"the previous tree\n")
    body = "prefix=tc/tt-metal@x\ndate() { echo 20260101-000000; }\nmove_aside previous\n"
    result = run_functions("tt-metal", ("move_aside",), body, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    aside = tmp_path / "tc" / "tt-metal@x.previous-20260101-000000"
    assert (aside / "kept.txt").read_bytes() == b"the previous tree\n"
    (tmp_path / "tc" / "tt-metal@x").mkdir()
    again = run_functions("tt-metal", ("move_aside",), body, tmp_path)
    assert again.returncode != 0 and "already exists" in again.stderr
    assert (tmp_path / "tc" / "tt-metal@x").is_dir() and (aside / "kept.txt").exists(), "nothing is lost"


@needs_bash
def test_expect_version_reads_the_first_line_of_a_tool(tmp_path: Path) -> None:
    stub = "tool() { printf 'cmake version 3.31.6\\n\\nCMake suite maintained\\n'; }\n"
    passes = run_functions("tt-metal", ("expect_version",), stub + "expect_version 'cmake version 3.31.6' tool\n",
                           tmp_path)
    assert passes.returncode == 0, passes.stderr
    fails = run_functions("tt-metal", ("expect_version",), stub + "expect_version 'cmake version 4.2.1' tool\n",
                          tmp_path)
    assert fails.returncode != 0 and "Decision Log" in fails.stderr


SPACE_STUBS = """du() { if [ -n "$DU_KIB" ]; then printf '%s\\t%s\\n' "$DU_KIB" "${@: -1}"; fi; }
df() { printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\\n/dev/x 1 1 %s 1%% /mnt\\n' "$DF_KIB"; }
LASSI_SCRATCH=/mnt/nvme10/scratch
"""
STOP_KIB = 115 * 1024 * 1024


@needs_bash
@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize(
    ("used", "free", "fits"),
    [(STOP_KIB - 1000, 10**9, True), (STOP_KIB - 999, 10**9, False), (1000, 999, False), ("", 10**9, False)],
    ids=["at-the-stop-line", "over-the-stop-line", "disk-full", "du-fails"],
)
def test_check_space_stops_at_the_plan_line(tmp_path: Path, name: str, used: int | str, free: int, fits: bool) -> None:
    body = SPACE_STUBS + 'check_space "1000"\n'
    result = run_functions(name, ("check_space",), body, tmp_path, env={"DU_KIB": str(used), "DF_KIB": str(free)})
    assert (result.returncode == 0) == fits, result.stdout + result.stderr


LIB = b"stand-in library bytes"
DESCRIPTOR = b"stand-in descriptor\n"
TTSIM_VALUES = (
    f"VERSION=v0\nURL=https://example.invalid/lib.so\nSIZE={len(LIB)}\nSHA256={sha(LIB)}\nLIBRARY=lib.so\n"
    f"SOC_DESCRIPTOR=soc_descriptor.yaml\nTT_METAL_FROM=tt-metal@x\nSOC_DESCRIPTOR_FROM=d.yaml\n"
    f"SOC_DESCRIPTOR_SHA256={sha(DESCRIPTOR)}\nrecord_name=lassi-install.txt\n"
)


def install_dir(path: Path, lib: bytes = LIB, record: bool = True) -> None:
    """Create a ttsim install at `path` from the stand-in files, with or without its record."""
    path.mkdir(parents=True)
    (path / "lib.so").write_bytes(lib)
    (path / "soc_descriptor.yaml").write_bytes(DESCRIPTOR)
    if record:
        lines = [
            f"ttsim v0 https://example.invalid/lib.so {len(LIB)} {sha(LIB)}",
            f"soc_descriptor soc_descriptor.yaml from tt-metal@x/d.yaml {sha(DESCRIPTOR)}",
        ]
        (path / "lassi-install.txt").write_bytes(("\n".join(lines) + "\n").encode("ascii"))


MATCHES = ("pinned_record", "has_sha256", "installed_matches")


@needs_bash
@pytest.mark.parametrize(
    ("lib", "record", "matches"),
    [(LIB, True, True), (b"stand-in library bytez", True, False), (LIB, False, False)],
    ids=["pinned", "other-bytes", "no-record"],
)
def test_ttsim_installed_matches_needs_record_size_and_hashes(
    tmp_path: Path, lib: bytes, record: bool, matches: bool
) -> None:
    install_dir(tmp_path / "i", lib=lib, record=record)
    result = run_functions("ttsim", MATCHES, TTSIM_VALUES + "installed_matches i\n", tmp_path)
    assert (result.returncode == 0) == matches, result.stdout + result.stderr


@needs_bash
def test_ttsim_swap_in_moves_the_previous_install_aside_and_keeps_it(tmp_path: Path) -> None:
    install_dir(tmp_path / "tc" / "ttsim@v0.staging")
    (tmp_path / "tc" / "ttsim@v0").mkdir()
    (tmp_path / "tc" / "ttsim@v0" / "old.txt").write_bytes(b"the previous install\n")
    body = (
        TTSIM_VALUES
        + 'prefix=tc/ttsim@v0\nstaging="$prefix.staging"\nmade_staging=1\ndate() { echo 20260101-000000; }\n'
        + 'swap_in\necho "aside=$aside made_staging=$made_staging"\ninstalled_matches "$prefix"\n'
    )
    result = run_functions("ttsim", (*MATCHES, "swap_in"), body, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "aside=tc/ttsim@v0.previous-20260101-000000 made_staging=0" in result.stdout
    assert (tmp_path / "tc" / "ttsim@v0.previous-20260101-000000" / "old.txt").exists(), "never deleted"
    assert not (tmp_path / "tc" / "ttsim@v0.staging").exists()


@pytest.mark.parametrize("name", NAMES)
def test_script_refuses_a_root_outside_the_scratch_root_before_anything(tmp_path: Path, name: str) -> None:
    # A path under /mnt/nvme10 but outside the scratch root, and one that only resolves outside it. The script
    # exits at the root check, before it makes a directory or runs a tool.
    assert BASH is not None
    for toolchains in ("/mnt/nvme10/other/toolchains", "/mnt/nvme10/joseph_ufl/../other"):
        env = {**os.environ, "LASSI_TOOLCHAINS": toolchains, "LASSI_SCRATCH": "/mnt/nvme10/joseph_ufl",
               "LASSI_JOBS": "4"}
        result = subprocess.run([BASH, f"toolchains/{name}.sh"], capture_output=True, text=True, cwd=REPO, env=env)
        assert result.returncode == 2, result.stdout + result.stderr
        assert "refusing to work outside /mnt/nvme10/joseph_ufl" in result.stderr
        assert result.stdout == ""


def test_tt_metal_script_clears_the_build_environment() -> None:
    text = script_text("tt-metal")
    unset = text[text.index("\nunset CC CXX ") :]
    unset = unset[: unset.index("\nfor name in $(compgen -e)")]
    names = set(unset.replace("\\\n", " ").split()) - {"unset"}
    required = {
        "CC", "CXX", "CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS", "CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH",
        "LIBRARY_PATH", "LD_LIBRARY_PATH", "CMAKE_PREFIX_PATH", "PKG_CONFIG_PATH", "CPM_USE_LOCAL_PACKAGES",
        "CPM_LOCAL_PACKAGES_ONLY",
    }
    assert required <= names, required - names
    settings = bash_function(text, "local_settings")
    assert "-DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF" in settings
    assert "-DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF" in settings


def sfpi_info_body(lines: str, status: int = 0) -> str:
    """Return a probe body with a stand-in tt_metal/sfpi-info.sh printing `lines`, then running the guard."""
    return (
        f'prefix="$PWD/t"\nSFPI_FILENAME=sfpi_7.25.0_x86_64_debian.txz\nSFPI_SHA256={SFPI_SHA256}\n'
        "mkdir -p t/tt_metal\n"
        f"printf '%s\\n' 'printf \"%s\\\\n\" \"$@\" >sfpi-info.args' 'cat <<\"EOF\"' {lines} 'EOF' "
        f"'exit {status}' >t/tt_metal/sfpi-info.sh\n"
        "check_sfpi_source\n"
    )


PINNED_SFPI_LINES = (
    "\"sfpi_version='7.25.0'\" \"sfpi_filename='sfpi_7.25.0_x86_64_debian.txz'\" "
    f"\"sfpi_hash='{SFPI_SHA256}'\""
)


@needs_bash
@pytest.mark.parametrize(
    ("lines", "status", "passes"),
    [
        (PINNED_SFPI_LINES, 0, True),
        (PINNED_SFPI_LINES.replace(SFPI_SHA256, "0" * 64), 0, False),
        (PINNED_SFPI_LINES.replace(f"sfpi_hash='{SFPI_SHA256}'", "sfpi_hash=''"), 0, False),
        (PINNED_SFPI_LINES.replace("x86_64_debian", "x86_64_unknown"), 0, False),
        (PINNED_SFPI_LINES, 1, False),
    ],
    ids=["pinned", "other-hash", "no-hash-for-host", "other-file", "script-fails"],
)
def test_check_sfpi_source_refuses_unless_the_lookup_names_the_pinned_txz(
    tmp_path: Path, lines: str, status: int, passes: bool
) -> None:
    result = run_functions("tt-metal", ("check_sfpi_source",), sfpi_info_body(lines, status), tmp_path)
    assert (result.returncode == 0) == passes, result.stdout + result.stderr
    assert (tmp_path / "sfpi-info.args").read_text().split() == ["SHELL", "txz"]


def linked_body(comment: str) -> str:
    """Return a probe body with a stand-in readelf printing `comment` for the gate's example."""
    return (
        "readelf() { printf '%s\\n' \"$@\" >readelf.args; printf '%b' \"$COMMENT\"; }\n"
        f"build=b\nEXAMPLES='{GATE_EXAMPLE} loopback'\nCLANG_EXPECT='clang version 20.1.8'\n"
        "LINKER_EXPECT='LLD 20.1.8'\ncheck_linked\n"
    )


LINKED = (
    "String dump of section '.comment':\\n  [     0]  Linker: Ubuntu LLD 20.1.8 (compatible with GNU linkers)\\n"
    "  [    37]  Ubuntu clang version 20.1.8 (++20250804090239+87f0227cb601-1~exp1~20250804210352.139)\\n"
    "  [    9e]  GCC: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0\\n"
)


@needs_bash
@pytest.mark.parametrize(
    ("comment", "passes"),
    [
        (LINKED, True),
        (LINKED.replace("Linker: Ubuntu LLD 20.1.8", "Linker: LLD 17.0.6"), False),
        (LINKED.replace("  [     0]  Linker: Ubuntu LLD 20.1.8 (compatible with GNU linkers)\\n", ""), False),
        (LINKED.replace("clang version 20.1.8", "clang version 17.0.6"), False),
    ],
    ids=["pinned", "other-lld", "gnu-ld", "other-clang"],
)
def test_check_linked_reads_the_compiler_and_linker_from_the_gate_example(
    tmp_path: Path, comment: str, passes: bool
) -> None:
    result = run_functions("tt-metal", ("check_linked",), linked_body(comment), tmp_path, env={"COMMENT": comment})
    assert (result.returncode == 0) == passes, result.stdout + result.stderr
    args = (tmp_path / "readelf.args").read_text().split()
    assert args == ["-p", ".comment", f"b/programming_examples/metal_example_{GATE_EXAMPLE}"]
