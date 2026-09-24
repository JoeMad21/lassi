"""Tests for the CUDA and NVHPC pin files and install scripts (P0.7, P0.19).

Pin files live in toolchains/<name>.pin and record the version, the install
sources, and the install path (bible Toolchain Pins). They are shell-sourced
KEY=value files, read by toolchains/<name>.sh, which installs in user space
under $LASSI_TOOLCHAINS/<name>@<version> (AGENTS.md Remote Execution) and is
idempotent. CUDA installs from NVIDIA's per-component redistributable
archives, each checked against the sha256 in NVIDIA's redistrib manifest,
whose own sha256 is pinned (OQ-010; plans/spikes/p0-cuda-redist.md); NVHPC
installs from its tarball. Most tests read the files only; the behavior tests
run single functions of toolchains/cuda.sh in bash on local files, with
stand-ins for curl, du, df, python3, find, id, mv, date, and nvcc where a test
needs one. The installs themselves run on the build
host through tools/rx.py. No value here is a measurement: EXPECT_VERSION holds
the version string the installed compiler must print, taken from the P0.6
spike (plans/spikes/p0-nvcc.md) for nvcc.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from lassi.toolchains.pins import PINS_DIR, read_pin

REPO = Path(__file__).resolve().parents[2]
TOOLCHAINS = REPO / "toolchains"
NAMES = ("cuda", "nvhpc")
REQUIRED_KEYS = ("NAME", "VERSION", "URL", "PREFIX_NAME", "EXPECT_VERSION")
REDIST = "https://developer.download.nvidia.com/compute/cuda/redist"
# The bash on PATH, resolved here: on Windows a bare "bash" given to subprocess can start the WSL launcher in
# System32 instead, which sees neither this environment nor Windows paths.
BASH = shutil.which("bash")
needs_bash = pytest.mark.skipif(BASH is None, reason="bash is not available")


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


def code_lines(text: str) -> list[str]:
    """Return the lines of a script that are not blank and not comments, stripped."""
    return [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]


def components() -> list[str]:
    """Return the component names toolchains/cuda.pin lists, in order."""
    return read_pin("cuda")["COMPONENTS"].split()


def archive_fields(component: str) -> tuple[str, str, str, str]:
    """Return (version, size, sha256, relative_path) from the pin's ARCHIVE_<COMPONENT> value."""
    fields = read_pin("cuda")[f"ARCHIVE_{component.upper()}"].split()
    assert len(fields) == 4, f"ARCHIVE_{component.upper()} must hold version, size, sha256, and relative path"
    version, size, sha256, path = fields
    return version, size, sha256, path


@pytest.mark.parametrize("name", NAMES)
def test_pin_records_version_sources_and_install_path(name: str) -> None:
    pin = read_pin(name)
    for key in REQUIRED_KEYS:
        assert pin.get(key), f"{name}.pin lacks {key}"
    assert pin["NAME"] == name
    assert pin["PREFIX_NAME"] == f"{name}@{pin['VERSION']}"
    assert pin["URL"].startswith("https://developer.download.nvidia.com/")


def test_cuda_pin_records_the_redist_manifest_and_its_checksum() -> None:
    pin = read_pin("cuda")
    assert pin["VERSION"] == "12.6.3"
    assert pin["PREFIX_NAME"] == "cuda@12.6.3"
    assert pin["EXPECT_VERSION"] == "Cuda compilation tools, release 12.6, V12.6.85"
    assert pin["URL"] == REDIST
    assert pin["MANIFEST_URL"] == f"{REDIST}/redistrib_{pin['VERSION']}.json"
    assert re.fullmatch(r"[0-9a-f]{64}", pin["MANIFEST_SHA256"])
    for retired in ("MD5", "INSTALL_FLAGS"):
        assert retired not in pin, f"cuda.pin still carries the runfile's {retired}"
    assert not pin["URL"].endswith(".run")


def test_cuda_pin_lists_each_component_with_its_version_size_and_sha256() -> None:
    pin = read_pin("cuda")
    names = components()
    assert len(names) == len(set(names)), "a component is listed twice"
    # nvcc with ptxas, nvlink, fatbinary, cicc, and libdevice; the runtime headers and libraries; and CCCL,
    # whose nv/target header the runtime's cuda_fp16.h includes (plans/spikes/p0-cuda-redist.md).
    assert {"cuda_nvcc", "cuda_cudart", "cuda_cccl"} <= set(names)
    listed = {f"ARCHIVE_{name.upper()}" for name in names}
    assert {key for key in pin if key.startswith("ARCHIVE_")} == listed, (
        "every archive key belongs to a listed component"
    )
    for name in names:
        version, size, sha256, path = archive_fields(name)
        assert re.fullmatch(r"\d+(\.\d+)+", version), name
        assert size.isdigit() and int(size) > 0, name
        assert re.fullmatch(r"[0-9a-f]{64}", sha256), name
        assert path == f"{name}/linux-x86_64/{name}-linux-x86_64-{version}-archive.tar.xz"
    # The nvcc archive's version is the compiler version EXPECT_VERSION names.
    assert f"V{archive_fields('cuda_nvcc')[0]}" in pin["EXPECT_VERSION"]


def test_cuda_pin_records_the_install_layout() -> None:
    pin = read_pin("cuda")
    assert pin["LAYOUT_PREFIX_DIRS"].split() == ["bin", "nvvm"]
    assert pin["LAYOUT_TARGET_DIR"] == "targets/x86_64-linux"
    assert pin["LAYOUT_TARGET_DIRS"].split() == ["include", "lib"]
    assert pin["LAYOUT_SKIP"].split() == ["pkg-config"], "only cuda_cudart's pkg-config/ is left out"
    links = dict(pair.split(":", 1) for pair in pin["LAYOUT_LINKS"].split())
    assert links == {"include": "targets/x86_64-linux/include", "lib64": "targets/x86_64-linux/lib"}
    for target in links.values():
        assert target.rsplit("/", 1)[1] in pin["LAYOUT_TARGET_DIRS"].split()
    assert pin["INSTALL_KIB"].isdigit() and int(pin["INSTALL_KIB"]) > 0


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


def test_cuda_script_keeps_an_install_only_when_its_version_and_component_record_match() -> None:
    text = script_text("cuda")
    skip = text.index("already holds the pinned")
    condition = text[text.rindex("\nif ", 0, skip) : skip]
    assert 'reports_pinned_version "$nvcc"' in condition
    assert '"$prefix/$record_name"' in condition and '"$(pinned_record)"' in condition
    assert 'pinned_record >"$staging/$record_name"' in text, "a new install writes the record the skip compares"


@pytest.mark.parametrize("name", NAMES)
def test_script_never_needs_root_or_system_paths(name: str) -> None:
    text = script_text(name)
    assert "sudo" not in text
    for system_path in ("/usr/local", "/opt/", "/etc/"):
        assert system_path not in text


@pytest.mark.parametrize("name", NAMES)
def test_script_verifies_the_download_and_the_installed_version(name: str) -> None:
    text = script_text(name)
    assert "sha256sum" in text
    assert "md5sum" not in text
    assert text.count("$EXPECT_VERSION") >= 2, "checked before skipping and after installing"


def test_cuda_script_checks_the_manifest_against_the_pinned_sha256_before_reading_it() -> None:
    text = script_text("cuda")
    fetch_manifest = text.index('fetch "$MANIFEST_URL" "$MANIFEST_SHA256" "$manifest"')
    read_entry = text.index('entry="$(manifest_entry "$c")"')
    assert fetch_manifest < read_entry
    entry = bash_function(text, "manifest_entry")
    for field in ('"linux-x86_64"', '"relative_path"', '"sha256"', '"size"', '"version"'):
        assert field in entry
    # The pin must agree with the manifest, so a pin edit cannot swap in an archive the manifest does not name.
    assert 'if [ "$entry" != "$spec" ]' in text


def test_cuda_script_checks_every_archive_against_its_manifest_sha256_before_extraction() -> None:
    text = script_text("cuda")
    loop = text[text.index('entry="$(manifest_entry "$c")"') :]
    loop = loop[: loop.index("\ndone\n")]
    assert 'read -r version size sum path <<<"$entry"' in loop, "the sha256 comes from the manifest entry"
    assert loop.index('fetch "$URL/$path" "$sum" "$archive"') < loop.index('extract_component "$c" "$archive"')
    assert bash_function(text, "fetch").count("sha256sum -c --status") == 2, "a reused file and a download"
    assert [line for line in code_lines(text) if line.startswith("tar -x")] == [
        'tar -xJf "$archive" -C "$dest" --strip-components=1 --keep-old-files "$top/$entry"'
    ], "extract_component holds the only extraction"


def test_cuda_script_refuses_when_the_planned_bytes_do_not_fit_before_any_download() -> None:
    text = script_text("cuda")
    space = bash_function(text, "check_space")
    assert 'du -sk "$LASSI_SCRATCH"' in space
    assert "120 * 1024 * 1024" in space, "the owner's 120 GiB cap on the scratch root"
    assert 'df -Pk "$LASSI_SCRATCH"' in space
    main = text[text.index("already holds the pinned") :]
    assert main.index('check_space "$planned_kib"') < main.index('fetch "$'), "checked before the first download"
    assert "planned_kib=$INSTALL_KIB" in main


def main_section(text: str) -> str:
    """Return the part of a script after its cleanup trap is set: the install steps, without the functions."""
    return text[text.index("\ntrap cleanup EXIT\n") :]


def test_cuda_script_stages_verifies_then_swaps_and_keeps_the_old_tree_aside() -> None:
    text = script_text("cuda")
    assert 'staging="$prefix.staging"' in text
    main = main_section(text)
    order = [
        'mkdir "$staging"',
        'extract_component "$c" "$archive"',
        'reports_pinned_version "$staging/bin/nvcc"',
        'check_compiles "$staging"',
        'check_root_fs "$marker" /tmp /var/tmp',
        "\nswap_in\n",
        'reports_pinned_version "$nvcc"',
    ]
    positions = [main.index(step) for step in order]
    assert positions == sorted(positions), "stage, verify, check the root filesystem, swap, check the swapped-in nvcc"
    assert "made_staging=1" in main[positions[0] :]
    swap = bash_function(text, "swap_in")
    assert 'aside="$prefix.$kind-$(date +%Y%m%d-%H%M%S)"' in swap
    assert swap.index('mv -T "$prefix" "$aside"') < swap.index('mv -T "$staging" "$prefix"')
    assert swap.index('mv -T "$staging" "$prefix"') < swap.index("made_staging=0")
    # Nothing is deleted that this run did not create: only its check directory, its marker, a download
    # it is writing, and its own staging prefix, which it removes only while the swap has not happened.
    removed = [line for line in code_lines(text) if re.search(r"\brm\b", line)]
    targets = {re.search(r'rm -r?f ("[^"]*")', line).group(1) for line in removed}
    assert targets == {'"$check"', '"$marker"', '"$part"', '"$staging"'}, removed
    assert 'if [ "$made_staging" = 1 ]; then rm -rf "$staging"; fi' in text


def test_cuda_script_runs_no_installer_and_writes_nothing_to_the_root_filesystem() -> None:
    text = script_text("cuda")
    for installer in (".run", "local_installers", "--toolkit", "--silent", "./install", 'sh "$'):
        assert installer not in text
    # The only command that names the system temporary directories is the check that nothing landed there.
    lines = [line for line in code_lines(text) if not line.startswith("echo ")]
    naming = [line for line in lines if re.search(r'(^|[\s"=(])/(var/)?tmp\b', line)]
    assert naming == ['check_root_fs "$marker" /tmp /var/tmp']
    assert not re.search(r'(^|[\s"=])/tmp\b', script_text("nvhpc")), "nvhpc.sh must not name the absolute /tmp"
    # That check only reads: find with no action that writes or runs anything, and no file of its own.
    check = bash_function(text, "check_root_fs")
    assert 'find "$@" -xdev -user "$(id -un)" -newer "$marker"' in check
    for writer in ("-delete", "-exec", "-fprint", "mktemp", "rm ", "mv ", "touch ", "mkdir "):
        assert writer not in check, writer
    # A find that could not look must not pass as a clean one: its errors are kept and read, not discarded.
    assert "2>/dev/null" not in check and "|| true" not in check
    assert "2>&1" in check


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


# Behavior tests: single functions of toolchains/cuda.sh, run in bash on local files.


def bash_function(text: str, name: str) -> str:
    """Return the definition of the top-level bash function `name`, from its `name() {` line to its `}` line."""
    match = re.search(rf"^{name}\(\) \{{\n.*?^\}}\n", text, re.S | re.M)
    assert match, f"cuda.sh defines no function {name}"
    return match.group(0)


def run_functions(
    names: tuple[str, ...], body: str, cwd: Path, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run `body` in strict bash in `cwd`, after the named functions of toolchains/cuda.sh."""
    text = script_text("cuda")
    script = "set -euo pipefail\n" + "".join(bash_function(text, name) for name in names) + body
    (cwd / "probe.sh").write_bytes(script.encode("ascii"))
    assert BASH is not None
    return subprocess.run(
        [BASH, "probe.sh"], cwd=cwd, capture_output=True, text=True, env={**os.environ, **(env or {})}
    )


# A stand-in for curl: copies served/<last URL segment> to the -o file and logs the URL.
CURL_STUB = """curl() {
  local out="" url=""
  while [ $# -gt 0 ]; do
    case "$1" in
      -o) out="$2"; shift 2 ;;
      --retry) shift 2 ;;
      -*) shift ;;
      *) url="$1"; shift ;;
    esac
  done
  echo "$url" >>curl.log
  cp "served/${url##*/}" "$out"
}
"""


def sha(data: bytes) -> str:
    """Return the sha256 hex digest of `data`."""
    return hashlib.sha256(data).hexdigest()


def fetch_setup(tmp_path: Path, served: bytes | None, existing: bytes | None) -> None:
    """Create served/a.tar.xz and downloads/a.tar.xz as given (None leaves a file out)."""
    (tmp_path / "served").mkdir()
    (tmp_path / "downloads").mkdir()
    if served is not None:
        (tmp_path / "served" / "a.tar.xz").write_bytes(served)
    if existing is not None:
        (tmp_path / "downloads" / "a.tar.xz").write_bytes(existing)


def run_fetch(tmp_path: Path, sum_: str) -> subprocess.CompletedProcess[str]:
    """Run fetch for https://example.invalid/r/a.tar.xz into downloads/a.tar.xz with a curl stand-in."""
    body = CURL_STUB + f"fetch https://example.invalid/r/a.tar.xz {sum_} downloads/a.tar.xz\n"
    return run_functions(("fetch",), body, tmp_path)


def part_files(tmp_path: Path) -> list[Path]:
    """Return the temporary download files left in downloads/."""
    return list((tmp_path / "downloads").glob("*.part*"))


@needs_bash
def test_fetch_downloads_checks_and_renames_a_missing_archive(tmp_path: Path) -> None:
    fetch_setup(tmp_path, served=b"archive bytes", existing=None)
    result = run_fetch(tmp_path, sha(b"archive bytes"))
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "downloads" / "a.tar.xz").read_bytes() == b"archive bytes"
    assert (tmp_path / "curl.log").read_text().split() == ["https://example.invalid/r/a.tar.xz"]
    assert part_files(tmp_path) == []


@needs_bash
def test_fetch_reuses_an_archive_with_the_expected_sha256(tmp_path: Path) -> None:
    fetch_setup(tmp_path, served=None, existing=b"archive bytes")
    result = run_fetch(tmp_path, sha(b"archive bytes"))
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "curl.log").exists(), "a verified download is not fetched again"


@needs_bash
def test_fetch_refuses_a_download_with_another_sha256_and_leaves_nothing(tmp_path: Path) -> None:
    fetch_setup(tmp_path, served=b"tampered bytes", existing=None)
    result = run_fetch(tmp_path, sha(b"archive bytes"))
    assert result.returncode != 0
    assert "sha256" in result.stderr
    assert not (tmp_path / "downloads" / "a.tar.xz").exists()
    assert part_files(tmp_path) == []


@needs_bash
def test_fetch_refuses_and_keeps_an_existing_file_with_another_sha256(tmp_path: Path) -> None:
    fetch_setup(tmp_path, served=b"archive bytes", existing=b"someone else's bytes")
    result = run_fetch(tmp_path, sha(b"archive bytes"))
    assert result.returncode != 0
    assert (tmp_path / "downloads" / "a.tar.xz").read_bytes() == b"someone else's bytes"
    assert not (tmp_path / "curl.log").exists()


# Stand-ins for du and df that report DU_KIB used under the scratch root and DF_KIB free on its disk.
SPACE_STUBS = """du() { if [ -n "$DU_KIB" ]; then printf '%s\\t%s\\n' "$DU_KIB" "${@: -1}"; fi; }
df() { printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\\n/dev/x 1 1 %s 1%% /mnt\\n' "$DF_KIB"; }
LASSI_SCRATCH=/mnt/nvme10/scratch
"""
CAP_KIB = 120 * 1024 * 1024


@needs_bash
@pytest.mark.parametrize(
    ("used", "planned", "free", "fits"),
    [
        (100 * 1024 * 1024, 250000, 400 * 1024 * 1024, True),
        (CAP_KIB - 250000, 250000, 400 * 1024 * 1024, True),
        (CAP_KIB - 250000 + 1, 250000, 400 * 1024 * 1024, False),
        (1000, 250000, 249999, False),
        ("", 250000, 400 * 1024 * 1024, False),
    ],
    ids=["well-under", "at-the-cap", "over-the-cap", "disk-full", "du-fails"],
)
def test_check_space_refuses_over_the_cap_or_the_free_space(
    tmp_path: Path, used: int | str, planned: int, free: int, fits: bool
) -> None:
    body = SPACE_STUBS + f'check_space "{planned}"\n'
    result = run_functions(("check_space",), body, tmp_path, env={"DU_KIB": str(used), "DF_KIB": str(free)})
    assert (result.returncode == 0) == fits, result.stdout + result.stderr
    if not fits:
        assert "cuda:" in result.stderr


def make_archive(path: Path, top: str, files: dict[str, bytes], outside: dict[str, bytes] | None = None) -> None:
    """Write an xz tar at `path` holding `files` under directory `top`, with directory members.

    `outside` maps member names to contents that are added as given, not
    under `top`, as an archive with a stray member would hold them.
    """
    with tarfile.open(path, "w:xz") as tar:
        dirs = {top}
        for name in files:
            parts = name.split("/")[:-1]
            dirs.update(f"{top}/" + "/".join(parts[: i + 1]) for i in range(len(parts)))
        for directory in sorted(dirs):
            info = tarfile.TarInfo(directory)
            info.type, info.mode = tarfile.DIRTYPE, 0o755
            tar.addfile(info)
        members = {f"{top}/{name}": data for name, data in files.items()} | (outside or {})
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o644
            tar.addfile(info, io.BytesIO(data))


LAYOUT = """staging=stage
LAYOUT_PREFIX_DIRS="bin nvvm"
LAYOUT_TARGET_DIR=targets/x86_64-linux
LAYOUT_TARGET_DIRS="include lib"
LAYOUT_SKIP="pkg-config"
mkdir -p "$staging/$LAYOUT_TARGET_DIR"
"""
EXTRACT = ("layout_dest", "layout_skips", "extract_component")


def extract(tmp_path: Path, *archives: tuple[str, str]) -> subprocess.CompletedProcess[str]:
    """Run extract_component for each (component, archive file) in order, into stage/."""
    calls = "".join(f"extract_component {component} {archive}\n" for component, archive in archives)
    return run_functions(EXTRACT, LAYOUT + calls, tmp_path)


@needs_bash
def test_extract_component_places_entries_by_the_layout_and_merges_components(tmp_path: Path) -> None:
    make_archive(
        tmp_path / "cuda_x-linux-x86_64-1.0.2-archive.tar.xz",
        "cuda_x-linux-x86_64-1.0.2-archive",
        {"bin/tool": b"t", "nvvm/bin/cicc": b"c", "include/a.h": b"a", "lib/liba.a": b"l", "LICENSE": b"x"},
    )
    make_archive(
        tmp_path / "cuda_y-linux-x86_64-1.0.2-archive.tar.xz",
        "cuda_y-linux-x86_64-1.0.2-archive",
        {"include/b.h": b"b", "include/sub/c.h": b"c", "LICENSE": b"y", "pkg-config/y.pc": b"p"},
    )
    result = extract(
        tmp_path,
        ("cuda_x", "cuda_x-linux-x86_64-1.0.2-archive.tar.xz"),
        ("cuda_y", "cuda_y-linux-x86_64-1.0.2-archive.tar.xz"),
    )
    assert result.returncode == 0, result.stderr
    assert "leaving cuda_y's pkg-config out of the install" in result.stdout
    stage = tmp_path / "stage"
    placed = sorted(p.relative_to(stage).as_posix() for p in stage.rglob("*") if p.is_file())
    assert placed == [
        "bin/tool",
        "licenses/cuda_x/LICENSE",
        "licenses/cuda_y/LICENSE",
        "nvvm/bin/cicc",
        "targets/x86_64-linux/include/a.h",
        "targets/x86_64-linux/include/b.h",
        "targets/x86_64-linux/include/sub/c.h",
        "targets/x86_64-linux/lib/liba.a",
    ]


@needs_bash
def test_extract_component_refuses_to_replace_a_file_another_component_placed(tmp_path: Path) -> None:
    make_archive(tmp_path / "cuda_x-1-archive.tar.xz", "cuda_x-1-archive", {"include/a.h": b"first"})
    make_archive(tmp_path / "cuda_z-1-archive.tar.xz", "cuda_z-1-archive", {"include/a.h": b"second"})
    result = extract(tmp_path, ("cuda_x", "cuda_x-1-archive.tar.xz"), ("cuda_z", "cuda_z-1-archive.tar.xz"))
    assert result.returncode != 0
    assert (tmp_path / "stage/targets/x86_64-linux/include/a.h").read_bytes() == b"first"


@needs_bash
@pytest.mark.parametrize(
    ("files", "outside"),
    [
        ({"bin/tool": b"t", "share/doc/readme": b"r"}, {}),  # a top-level entry the layout does not place
        ({"bin/tool": b"t"}, {"other-archive/bin/extra": b"o"}),  # a member outside the archive's top directory
    ],
    ids=["unplaced-entry", "outside-member"],
)
def test_extract_component_refuses_an_archive_the_layout_cannot_hold_before_extracting(
    tmp_path: Path, files: dict[str, bytes], outside: dict[str, bytes]
) -> None:
    make_archive(tmp_path / "cuda_w-1-archive.tar.xz", "cuda_w-1-archive", files, outside)
    result = extract(tmp_path, ("cuda_w", "cuda_w-1-archive.tar.xz"))
    assert result.returncode != 0
    assert "cuda:" in result.stderr
    assert not (tmp_path / "stage" / "bin").exists(), "nothing is extracted from a refused archive"


@needs_bash
def test_manifest_entry_prints_the_pin_fields_for_each_component(tmp_path: Path) -> None:
    # A manifest built from the pin's values must give back each ARCHIVE_ value exactly, so the script's
    # comparison of the pin with NVIDIA's manifest checks every field in the same order.
    manifest: dict[str, object] = {"release_label": "12.6.3"}
    for name in components():
        version, size, sha256, path = archive_fields(name)
        manifest[name] = {
            "version": version,
            "linux-x86_64": {"relative_path": path, "sha256": sha256, "md5": "0" * 32, "size": size},
        }
    (tmp_path / "m.json").write_text(json.dumps(manifest), encoding="utf-8")
    python = Path(sys.executable).as_posix()
    calls = "".join(f"manifest_entry {name}\n" for name in components())
    body = f'python3() {{ "{python}" "$@"; }}\nmanifest=m.json\n' + calls
    result = run_functions(("manifest_entry",), body, tmp_path)
    assert result.returncode == 0, result.stderr
    pin = read_pin("cuda")
    assert result.stdout.splitlines() == [pin[f"ARCHIVE_{name.upper()}"] for name in components()]


@needs_bash
def test_bash_reads_the_cuda_pin_as_read_pin_does(tmp_path: Path) -> None:
    pin = read_pin("cuda")
    shutil.copy(TOOLCHAINS / "cuda.pin", tmp_path / "cuda.pin")
    body = "source ./cuda.pin\n" + "".join(f'printf "%s\\n" "${key}"\n' for key in pin)
    result = run_functions((), body, tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == list(pin.values())


# check_root_fs: stand-ins for find, which logs its arguments and prints FIND_OUT, FIND_ERR (to stderr), and
# exits with FIND_RC, and for id. The directories are real ones under the test's directory.
FIND_STUBS = """find() {
  echo "$*" >>find.log
  printf '%b' "$FIND_OUT"
  printf '%b' "$FIND_ERR" >&2
  return "$FIND_RC"
}
id() { echo tester; }
mkdir -p sys/tmp sys/var/tmp
"""
DENIED = "find: 'sys/tmp/systemd-private-x': Permission denied\\n"


def run_check_root_fs(
    tmp_path: Path, out: str, err: str, rc: int, dirs: str = "sys/tmp sys/var/tmp"
) -> subprocess.CompletedProcess[str]:
    """Run check_root_fs over `dirs` with a find stand-in that prints `out` and `err` and exits with `rc`."""
    body = FIND_STUBS + f"check_root_fs start.marker {dirs}\n"
    env = {"FIND_OUT": out, "FIND_ERR": err, "FIND_RC": str(rc)}
    return run_functions(("check_root_fs",), body, tmp_path, env=env)


@needs_bash
@pytest.mark.parametrize(
    ("err", "rc", "skipped"),
    [("", 0, "0 unreadable"), (DENIED + DENIED, 1, "2 unreadable")],
    ids=["clean", "permission-denied-only"],
)
def test_check_root_fs_passes_when_find_ran_and_listed_nothing(tmp_path: Path, err: str, rc: int, skipped: str) -> None:
    result = run_check_root_fs(tmp_path, out="", err=err, rc=rc)
    assert result.returncode == 0, result.stdout + result.stderr
    assert skipped in result.stdout, "the pass says how many paths find could not read"
    args = (tmp_path / "find.log").read_text().split()
    assert args[:2] == ["sys/tmp", "sys/var/tmp"]
    assert "-xdev" in args and args[args.index("-user") + 1] == "tester"
    assert args[args.index("-newer") + 1] == "start.marker"


@needs_bash
def test_check_root_fs_fails_and_lists_a_file_changed_during_the_install(tmp_path: Path) -> None:
    result = run_check_root_fs(tmp_path, out="changed sys/tmp/nvacc1.bc\\n", err=DENIED, rc=1)
    assert result.returncode != 0
    assert "sys/tmp/nvacc1.bc" in result.stderr
    assert "nothing is swapped in" in result.stderr


@needs_bash
@pytest.mark.parametrize(
    ("err", "rc"),
    [
        ("find: 'sys/tmp': No such file or directory\\n", 1),  # an error other than permission denied
        (DENIED + "find: 'sys/var/tmp/x': Input/output error\\n", 1),  # one other error among the denied ones
        ("", 1),  # a failure with no message at all
        ("find: invalid argument `-user'\\n", 1),  # find refused its arguments and looked nowhere
    ],
    ids=["missing-start-point", "io-error", "silent-failure", "bad-arguments"],
)
def test_check_root_fs_fails_when_find_could_not_look(tmp_path: Path, err: str, rc: int) -> None:
    result = run_check_root_fs(tmp_path, out="", err=err, rc=rc)
    assert result.returncode != 0, "a find that could not look is not a clean check"
    assert "did not run cleanly" in result.stderr


@needs_bash
def test_check_root_fs_refuses_a_directory_that_is_not_there_before_running_find(tmp_path: Path) -> None:
    result = run_check_root_fs(tmp_path, out="", err="", rc=0, dirs="sys/tmp sys/no-such-dir")
    assert result.returncode != 0
    assert "sys/no-such-dir" in result.stderr
    assert not (tmp_path / "find.log").exists()


# swap_in: runs on local directories under tc/, with the pinned record name and a stand-in nvcc.
SWAP = """prefix=tc/cuda@12.6.3
staging="$prefix.staging"
record_name=redist-components.txt
EXPECT_VERSION="Cuda compilation tools, release 12.6, V12.6.85"
made_staging=1
aside=""
"""
SWAP_REPORT = """swap_in
echo "aside=$aside made_staging=$made_staging"
reports_pinned_version "$prefix/bin/nvcc"
echo "swapped-in nvcc reports the pinned version"
"""
STAND_IN_NVCC = b'#!/bin/sh\necho "Cuda compilation tools, release 12.6, V12.6.85"\n'


def swap_setup(tmp_path: Path, old: str | None) -> Path:
    """Create a staging prefix under tc/ and, unless `old` is None, a previous install ("runfile" or "redist")."""
    tc = tmp_path / "tc"
    staging = tc / "cuda@12.6.3.staging"
    (staging / "bin").mkdir(parents=True)
    (staging / "bin" / "nvcc").write_bytes(STAND_IN_NVCC)
    (staging / "bin" / "nvcc").chmod(0o755)
    (staging / "redist-components.txt").write_bytes(b"new record\n")
    if old is not None:
        (tc / "cuda@12.6.3").mkdir()
        (tc / "cuda@12.6.3" / "old.txt").write_bytes(b"the previous install\n")
        if old == "redist":
            (tc / "cuda@12.6.3" / "redist-components.txt").write_bytes(b"old record\n")
    return tc


def run_swap(tmp_path: Path, stubs: str = "", report: str = SWAP_REPORT) -> subprocess.CompletedProcess[str]:
    """Run swap_in after the SWAP settings and any stand-ins, then print its results and check the nvcc."""
    return run_functions(("reports_pinned_version", "swap_in"), SWAP + stubs + report, tmp_path)


def entries(tc: Path) -> list[str]:
    """Return the names under tc/, sorted."""
    return sorted(p.name for p in tc.iterdir())


@needs_bash
def test_swap_in_renames_staging_into_place_for_a_first_install(tmp_path: Path) -> None:
    tc = swap_setup(tmp_path, old=None)
    result = run_swap(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "aside= made_staging=0" in result.stdout
    assert "swapped-in nvcc reports the pinned version" in result.stdout
    assert entries(tc) == ["cuda@12.6.3"]
    assert (tc / "cuda@12.6.3" / "redist-components.txt").read_bytes() == b"new record\n"


@needs_bash
@pytest.mark.parametrize("old", ["runfile", "redist"])
def test_swap_in_moves_a_previous_install_aside_and_keeps_it(tmp_path: Path, old: str) -> None:
    tc = swap_setup(tmp_path, old=old)
    result = run_swap(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    names = entries(tc)
    assert len(names) == 2 and names[0] == "cuda@12.6.3", names
    assert re.fullmatch(rf"cuda@12\.6\.3\.{old}-\d{{8}}-\d{{6}}", names[1]), "an install with a record is redist"
    assert f"aside=tc/{names[1]} made_staging=0" in result.stdout
    assert (tc / names[1] / "old.txt").read_bytes() == b"the previous install\n", "the old tree is kept whole"
    assert (tc / "cuda@12.6.3" / "redist-components.txt").read_bytes() == b"new record\n"
    assert "swapped-in nvcc reports the pinned version" in result.stdout


@needs_bash
def test_swap_in_puts_the_previous_install_back_when_the_rename_fails(tmp_path: Path) -> None:
    tc = swap_setup(tmp_path, old="runfile")
    stub = 'mv() { if [ "$2" = "$staging" ]; then echo "mv: stand-in failure" >&2; return 1; fi; command mv "$@"; }\n'
    result = run_swap(tmp_path, stubs=stub, report='swap_in || echo "swap_in rc=$? made_staging=$made_staging"\n')
    assert "swap_in rc=1 made_staging=1" in result.stdout, result.stdout + result.stderr
    assert "back in place" in result.stderr
    assert entries(tc) == ["cuda@12.6.3", "cuda@12.6.3.staging"], "no tree is left aside or lost"
    assert (tc / "cuda@12.6.3" / "old.txt").read_bytes() == b"the previous install\n"


@needs_bash
def test_swap_in_leaves_the_previous_install_aside_when_it_cannot_be_put_back(tmp_path: Path) -> None:
    tc = swap_setup(tmp_path, old="runfile")
    stub = """mv() {
  case "$2" in
    "$staging" | *.runfile-*) echo "mv: stand-in failure" >&2; return 1 ;;
  esac
  command mv "$@"
}
"""
    result = run_swap(tmp_path, stubs=stub, report='swap_in || echo "swap_in rc=$?"\n')
    assert "swap_in rc=1" in result.stdout, result.stdout + result.stderr
    names = entries(tc)
    assert len(names) == 2 and names[1] == "cuda@12.6.3.staging", names
    aside = names[0]
    assert aside.startswith("cuda@12.6.3.runfile-")
    assert (tc / aside / "old.txt").read_bytes() == b"the previous install\n", "the old tree is never deleted"
    assert f"tc/{aside}" in result.stderr, "the message says where the previous install is"


@needs_bash
def test_swap_in_refuses_when_the_aside_name_is_taken(tmp_path: Path) -> None:
    tc = swap_setup(tmp_path, old="runfile")
    (tc / "cuda@12.6.3.runfile-20260101-000000").mkdir()
    (tc / "cuda@12.6.3.runfile-20260101-000000" / "keep.txt").write_bytes(b"not this run's\n")
    stub = "date() { echo 20260101-000000; }\n"
    result = run_swap(tmp_path, stubs=stub, report='swap_in || echo "swap_in rc=$?"\n')
    assert "swap_in rc=1" in result.stdout, result.stdout + result.stderr
    assert "already exists" in result.stderr
    assert entries(tc) == ["cuda@12.6.3", "cuda@12.6.3.runfile-20260101-000000", "cuda@12.6.3.staging"]
    assert (tc / "cuda@12.6.3" / "old.txt").read_bytes() == b"the previous install\n"
    assert (tc / "cuda@12.6.3.runfile-20260101-000000" / "keep.txt").read_bytes() == b"not this run's\n"
