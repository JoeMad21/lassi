"""Tests for tools/local/install-batch1.sh, the workstation's first install batch (task P5.15).

Bible: Host Facts (Workstation, Workstation scratch, Workstation approvals),
Toolchain Pins (the pins stay), Agent Rules 7 and 10; AGENTS.md Local
Execution; plans/p5-ir.md P5.15 ("First install batch"); plans/local-plan.md
section 2.3 and section 7, decision 2 (b). The owner reviews and runs the
script, and no test runs a stage.

The contract these tests fix, for the implementation to follow:

- One bash script, `#!/usr/bin/env bash` and `set -euo pipefail`, plain ASCII
  with LF line endings, with two stages:
  - `bash tools/local/install-batch1.sh import --disk-dir <dir> --download-dir
    <dir>`, from Git Bash on Windows: refuses a disk dir that is not on C: or
    Z:, refuses when the distro LASSI-Ubuntu-22.04 already exists (`wsl.exe -l
    -q`), fetches the pinned Ubuntu 22.04 WSL image into the download dir,
    checks its size and sha256, and runs `wsl.exe --import LASSI-Ubuntu-22.04
    <disk dir> <image> --version 2`.
  - `wsl.exe -d LASSI-Ubuntu-22.04 -u root --exec bash
    /mnt/c/dev/lassi/tools/local/install-batch1.sh packages --user <name>
    --root <scratch root>`, inside the distro as root: never sudo; creates the
    user (useradd -m -s /bin/bash, no password, no sudo group) before any
    directory under the scratch root exists, then the scratch root; logs to
    <root>/install/batch1/install.log through tee, the user's creation
    included; writes /etc/wsl.conf ([boot] systemd=true, [user]
    default=<name>); checks apt.llvm.org's key in a GnuPG home it makes under
    <root>/install/batch1 and removes once the key is dearmored; downloads the
    .debs first into <root>/install/batch1/debs (apt-get --download-only with
    -o Dir::Cache::archives=...), checks each named .deb's size and sha256,
    then installs every package as name=version, with --no-remove on every
    apt-get install; unpacks CMake into /usr/local; puts uv in <root>/bin;
    logs dpkg-query -W; gives <root>/install to the user (chown -R) after the
    last step that writes under the root, and reads du -sk of the root back as
    the user; and compares each tool's version line with the string its pin
    file holds, read at run time (gcc.pin, tt-metal.pin, tt-mlir.pin), then
    tells the owner to run `wsl.exe --terminate LASSI-Ubuntu-22.04`.
- The header, the text before the first function, holds only comments, `set`
  options, and `readonly NAME=value` statements (a multi-line value as
  `readonly -a NAME=(` ... `)` or `readonly NAME="` ... `"`), with comments
  naming each index or release page and the date read, 2026-10-17. Its names:
  DISTRO (LASSI-Ubuntu-22.04); ROOTFS_URL, ROOTFS_SIZE, ROOTFS_SHA256;
  LLVM_KEY_URL, LLVM_KEY_SIZE, LLVM_KEY_SHA256, and optionally
  LLVM_KEY_FINGERPRINT (40 hex digits, no spaces); CMAKE_VERSION, CMAKE_URL,
  CMAKE_SIZE, CMAKE_SHA256; UV_VERSION, UV_URL, UV_SIZE, UV_SHA256; and
  PACKAGES, one entry per line, `<name>=<version> <deb size> <deb sha256>`,
  for exactly the named packages. Every NAME_URL is a download and has a
  NAME_SIZE and a NAME_SHA256 (an apt repository's address takes another name).
- Top-level functions in toolchains/*.sh style (`name() {` through a `}` line
  at column 0). Messages go to stderr and start with `install-batch1:`.
  - parse_import_args "$@": sets disk_dir and download_dir; refuses a missing
    or empty value, an option with no value, and an unknown argument, naming it.
  - parse_packages_args "$@": sets user_name and scratch_root, the same way.
    Both parse functions only read their arguments: no check, no file.
  - check_disk_dir <dir>: accepts C: or Z: (C:\\..., C:/..., /c/..., any
    case) and refuses everything else, judging the text alone.
  - check_no_distro: runs `wsl.exe -l -q` and nothing else, and refuses when
    one line of its output is exactly LASSI-Ubuntu-22.04. wsl.exe writes that
    list in UTF-16LE with CRLF line ends unless WSL_UTF8=1 is in its
    environment, when it writes UTF-8.
  - check_scratch_root <root>: refuses a relative root, "/", and a root that
    lies under /mnt, /tmp, /var/tmp, or /opt once `..` is resolved. It is the
    only place the script names /tmp, /var/tmp, or /opt.
  - check_euid: refuses unless `id -u` prints 0 (id, not $EUID, so a test can
    stand in for it).
  - check_file <path> <size> <sha256>: returns 0 only when the file exists
    with that size in bytes and that sha256.
  - packages_stage: runs check_packages_host, then ensure_user (useradd and
    the user checks, kept for the log), then makes the directories under the
    root, then run_packages through tee; run_packages's steps run in the order
    the probes below name, own_root (which logs what ensure_user did) right
    after record_host start.
  - install_llvm_key <keyring>: checks and dearmors the key with GNUPGHOME a
    directory from mktemp -d <root>/install/batch1/gnupg.XXXXXX, then removes
    it with remove_gnupg_home.
  - remove_gnupg_home <dir>: removes only a directory named gnupg. and six
    letters or digits directly in <root>/install/batch1, with root absolute,
    and refuses any other path.
  - give_install_tree: chown -R <user>:<group> <root>/install, then
    runuser -u <user> -- du -sk <root>.

No test runs a stage. The probes run the header and the functions in the bash
on PATH, with PATH cut to /usr/bin:/bin (where Git Bash keeps no wsl.exe and
no curl) and stand-ins, as bash functions or logging scripts, for wsl.exe,
wsl, curl, wget, apt-get, apt, apt-key, useradd, usermod, adduser, passwd,
chpasswd, dpkg, dpkg-query, systemd-run, prlimit, sudo, runuser, chown, and
id. The whole
script runs only with no stage, an unknown stage, or a stage without its
arguments, which it must refuse before it runs anything. The pinned values
below were read from the published indexes and release pages on 2026-10-17
(each source named); they are index reads, not measurements. Fixture files
are SYNTHETIC.
"""

from __future__ import annotations

import hashlib
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

from lassi.toolchains.pins import read_pin

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "tools" / "local" / "install-batch1.sh"
SCRIPT_REL = "tools/local/install-batch1.sh"
CPU_TESTS = REPO / ".github" / "workflows" / "cpu-tests.yml"
# The bash on PATH, resolved here: on Windows a bare "bash" given to subprocess can start the WSL launcher in
# System32 instead, which sees neither this environment nor Windows paths (tests/toolchains/test_tt_pins.py).
BASH = shutil.which("bash")
needs_bash = pytest.mark.skipif(BASH is None, reason="bash is not available")
DISTRO = "LASSI-Ubuntu-22.04"
PREFIX = "install-batch1:"
FUNCTION = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\(\) \{\n.*?^\}\n", re.S | re.M)
READONLY = re.compile(r"^readonly (?:-a )?([A-Z][A-Z0-9_]*)=", re.M)
HEX64 = re.compile(r"[0-9a-f]{64}")
HEX40 = re.compile(r"[0-9A-F]{40}")
PUNCTUATION = set("();<>|&")

# The Ubuntu 22.04 WSL images, read 2026-10-17. Either may be pinned; the script names one.
# - https://releases.ubuntu.com/22.04.5/SHA256SUMS (the same file is served under /jammy/), Last-Modified
#   2026-03-09: the current official Ubuntu 22.04 WSL image (https://cloud-images.ubuntu.com/wsl/ says that
#   publication moved).
# - https://cloud-images.ubuntu.com/wsl/releases/22.04/20240304/SHA256SUMS: the last dated jammy release there.
UBUNTU_WSL_SHA256 = "4499c4fe257f2fc83145b429ce211a0a43fd590e70d6261ede616210947d9f8f"
ROOTFS_CHOICES = {
    "https://releases.ubuntu.com/22.04.5/ubuntu-22.04.5-wsl-amd64.wsl": (360684292, UBUNTU_WSL_SHA256),
    "https://releases.ubuntu.com/jammy/ubuntu-22.04.5-wsl-amd64.wsl": (360684292, UBUNTU_WSL_SHA256),
    "https://cloud-images.ubuntu.com/wsl/releases/22.04/20240304/ubuntu-jammy-wsl-amd64-wsl.rootfs.tar.gz": (
        234074612,
        "de9f6149da07b90350a3ccd94b4858b82fef71f0ec2982acb93de583c4c87585",
    ),
}
# apt.llvm.org's signing key, read 2026-10-17: 3167 bytes; its primary key's fingerprint (gpg --show-keys) is the
# Debian LLVM packages key.
LLVM_KEY = (
    "https://apt.llvm.org/llvm-snapshot.gpg.key",
    3167,
    "8b2a587ffd672c4687e7581dad4b2f6c1bb2ad6b480cd9771ba2ff48e0b8c75d",
)
LLVM_KEY_FINGERPRINT = "6084F3CF814B57C1CF12EFD515CF4D18AF4F7421"
# Kitware's release page, cmake-3.31.6-SHA-256.txt, and the asset's Content-Length, read 2026-10-17.
CMAKE = (
    "https://github.com/Kitware/CMake/releases/download/v3.31.6/cmake-3.31.6-linux-x86_64.tar.gz",
    55010959,
    "5a1133ff103c71eb5120e2cc3de922733e7d8a26a98ae716397e8676adb367bf",
)
# uv's release page, uv-x86_64-unknown-linux-gnu.tar.gz.sha256, and the asset's Content-Length, read 2026-10-17.
UV = (
    "https://github.com/astral-sh/uv/releases/download/0.12.17/uv-x86_64-unknown-linux-gnu.tar.gz",
    19755224,
    "fa82fd8dde8e8eefdecada6aa0889666556cfceb690d06e0c3bca49eb3070a63",
)
# The named packages: version, .deb size, and .deb sha256, read 2026-10-17 from
# https://apt.llvm.org/jammy/dists/llvm-toolchain-jammy-{17,20}/main/binary-amd64/Packages and
# http://archive.ubuntu.com/ubuntu/dists/{jammy,jammy-updates,jammy-security}/{main,universe}/binary-amd64/Packages.gz,
# each the highest version served (curl's is jammy-security's; jammy-updates serves 7.81.0-1ubuntu1.29).
LLVM17 = "1:17.0.6~++20231209124227+6009708b4367-1~exp1~20231209124336.77"
LLVM20 = "1:20.1.8~++20250708082409+6fb913d3e2ec-1~exp1~20250708202428.132"
PY310 = "3.10.12-1~22.04.18"
GCC12 = "12.3.0-1ubuntu1~22.04.3"
PACKAGES = {
    "clang-17": (LLVM17, 144342, "8fc71a5739bfe76d590ff282740ac73735dbf7ac0d1c1c7b3da84628f07d9690"),
    "lld-17": (LLVM17, 1367962, "1086e1d6d20d52cb1b6da01812a6fdccab80d8b3efb01dcd10bde7a63c2887e9"),
    "clang-20": (LLVM20, 151152, "51bdb50d21a3ef8589afc777f94d7c24331ee4d646e13dcff7b1dbad00113dc4"),
    "lld-20": (LLVM20, 1562264, "ecad131aabee8b1e76affc788e363bf6e58a4e7584b5af68ee1b926960c2fd21"),
    "g++-12": (GCC12, 12185324, "c4123ea47a1d422ec43fb0918f670cfeb7bef1f26e6bf372fe15099fd42bbfcd"),
    "ninja-build": ("1.10.1-1", 111368, "ca8aae8ac7efad41d1f96a7e227653bd62b24e89cc7ddb18cec262f1869877b2"),
    "python3.10": (PY310, 508494, "10b62fd31bd0310a0b6391c04b67a1f6b33d180820b4a6cf7d25b55b993a15d7"),
    "python3.10-venv": (PY310, 5724, "fb7d9e9e10de015da2eab2a6da33fc6672d096b9281ccac0ecf990de4f724407"),
    "python3.10-dev": (PY310, 507980, "7caa163f4d83ae871f985ec4339fb4c96ae7ef8cb3568b8aec9138f86aacbeb7"),
    "git": ("1:2.34.1-1ubuntu1.17", 3173622, "8794fcf2c4606c445df0db3dc963c8fb852772208bfb12727a12717c03767af7"),
    "curl": ("7.81.0-1ubuntu1.30", 194114, "5c59f709ee58e2b12ec9c294b62e2e8cce8f096aa267014e34985cf1960a892f"),
    "make": ("4.3-4.1build1", 179742, "080b79a1a1623a2e6c6eead37d62b15fdf2c3dbfeafe8ecf5e31c54eb09eadcc"),
    "pkg-config": ("0.29.2-1ubuntu3", 48218, "69da4190d948d0bf7444ead040791b7924bd1a08a10ec13cd7f0bb9baab09c53"),
    "libhwloc-dev": ("2.7.0-2ubuntu1", 255576, "a097a8e4596eb3d40a1f069ceab00ea280c527bcd9917fe3f5ae87ec5cdc1b88"),
    "libnuma-dev": ("2.0.14-3ubuntu2", 35872, "d127bf32ebc3459645bd07fb18dd24f5780f405009589383c533f97d2727643b"),
    "libtbb-dev": ("2021.5.0-7ubuntu2", 190718, "d3052b9b07c59315cc87d18f7a9ed9c3c5d8d7c446104cdcba6bd17ddfb1fbaa"),
    "libcapstone-dev": ("4.0.2-5", 794936, "c13611a1dd1a184827cb4d9eccd573522fa59f3ea75bd5c22e5431adeb6b2a6d"),
    "libssl-dev": ("3.0.2-0ubuntu1.30", 2377254, "8a47dfa7f9bd4d54ac9dc9a1db14891a9133841bf44f12c3db086f2d4321bac6"),
    "xz-utils": ("5.2.5-2ubuntu1.1", 84826, "012652d4f76a08bfc6badf6a37c6da4c4e4a9f78b8c1560951750a74d700405c"),
}
# Commands no probe may reach: each gets a stand-in that logs its arguments and does nothing else.
STUBBED = (
    "wsl.exe", "wsl", "curl", "wget", "apt-get", "apt", "apt-key", "useradd", "usermod", "adduser", "passwd",
    "chpasswd", "dpkg", "dpkg-query", "systemd-run", "prlimit", "sudo", "runuser", "chown",
)
PLAIN_STUBS = tuple(name for name in STUBBED if not name.startswith("wsl"))


def log_stub(name: str) -> str:
    """Return a bash function `name` that logs its name and arguments to $STUB_LOG and does nothing else."""
    return f'{name}() {{ printf "%s\\n" "{name} $*" >> "$STUB_LOG"; }}\n'


# The stand-ins as bash functions, defined after the script's functions so they win. The wsl.exe one prints the
# distro list a test wrote, in UTF-8 when WSL_UTF8=1 reaches it and in UTF-16LE otherwise, as wsl.exe does.
STUB_FUNCTIONS = "".join(log_stub(name) for name in PLAIN_STUBS) + (
    """wsl.exe() {
  printf '%s\\n' "wsl.exe $*" >> "$STUB_LOG"
  if [ "${WSL_UTF8:-}" = 1 ]; then cat "$WSL_LIST_DIR/utf8.txt"; else cat "$WSL_LIST_DIR/utf16.txt"; fi
}
wsl() { wsl.exe "$@"; }
id() { if [ "${1:-}" = -u ]; then printf '%s\\n' "${FAKE_UID:-1000}"; else command id "$@"; fi; }
"""
)
# The stand-ins as scripts, for the whole-script probe.
STUB_SCRIPT = '#!/bin/sh\nprintf "%s\\n" "{name} $*" >> "$STUB_LOG"\nexit 0\n'
# Words after which a word is in command position.
COMMAND_BEFORE = {"|", "||", "&&", ";", "(", "$(", "then", "do", "else", "exec", "env", "xargs", "nice", "command"}


# Helpers.


def script_text() -> str:
    """Return the text of tools/local/install-batch1.sh; fail plainly while it does not exist."""
    assert SCRIPT.is_file(), f"{SCRIPT_REL} does not exist yet"
    return SCRIPT.read_bytes().decode("ascii")


def header(text: str) -> str:
    """Return the text before the script's first top-level function."""
    match = FUNCTION.search(text)
    assert match, "the script defines no function in the name() { ... } style"
    return text[: match.start()]


def functions(text: str) -> dict[str, str]:
    """Return every top-level function of the script, its name mapped to its definition."""
    return {match.group(1): match.group(0) for match in FUNCTION.finditer(text)}


def code_lines(text: str) -> list[str]:
    """Return the stripped code lines of bash text, continuations joined, without comment or blank lines."""
    joined = re.sub(r"\\\n\s*", " ", text)
    lines = [line.strip() for line in joined.splitlines()]
    return [line for line in lines if line and not line.startswith("#")]


def tokens(line: str) -> list[str]:
    """Return the shell words and operators of one code line, comments dropped; [] when it does not split."""
    lexer = shlex.shlex(line, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    try:
        return list(lexer)
    except ValueError:
        return []


def in_command_position(words: list[str], index: int) -> bool:
    """Return whether words[index] is where a command name stands."""
    return index == 0 or words[index - 1] in COMMAND_BEFORE or set(words[index - 1]) <= PUNCTUATION


def write_probe(cwd: Path, body: str) -> None:
    """Write `body` as probe.sh in `cwd`, ASCII with LF line ends."""
    (cwd / "probe.sh").write_bytes(body.encode("ascii"))


def run_bash(cwd: Path, body: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Run `body` as probe.sh in bash in `cwd`; the probe sets its own PATH, so no Windows program is reachable."""
    write_probe(cwd, body)
    assert BASH is not None
    return subprocess.run(
        [BASH, "probe.sh"], cwd=cwd, capture_output=True, text=True, env={**os.environ, **(env or {})}, timeout=60
    )


def probe_prelude(cwd: Path) -> str:
    """Return the probe lines that cut PATH, name the stand-in log, and point the wsl.exe stand-in at its lists."""
    lists = cwd / "lists"
    lists.mkdir(exist_ok=True)
    for name in ("utf8.txt", "utf16.txt"):
        if not (lists / name).exists():
            (lists / name).write_bytes(b"")
    return (
        "set -euo pipefail\nexport PATH=/usr/bin:/bin\n"
        'export STUB_LOG="$PWD/stub.log"\n: > "$STUB_LOG"\nexport WSL_LIST_DIR="$PWD/lists"\n'
    )


def run_functions(cwd: Path, body: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Run `body` after the script's header, its functions, and the stand-ins (never its main section)."""
    text = script_text()
    script_part = header(text) + "".join(functions(text).values())
    return run_bash(cwd, probe_prelude(cwd) + script_part + STUB_FUNCTIONS + body, env)


def stub_calls(cwd: Path) -> list[str]:
    """Return the stand-in calls a probe logged."""
    path = cwd / "stub.log"
    return path.read_text(encoding="ascii").splitlines() if path.exists() else []


HEADER_EMIT = """__probe_emit() {
  local IFS=$'\\n' __probe_decl
  __probe_decl="$(declare -p "$1")"
  local -n __probe_ref="$1"
  printf '%s\\0%s\\0%s\\0' "$1" "${__probe_decl%%=*}" "${__probe_ref[*]}"
}
"""


def header_values(tmp_path: Path) -> dict[str, tuple[str, str]]:
    """Return each readonly name of the header mapped to (its declare words, its value as bash evaluates it).

    An array's elements are joined with newlines, so PACKAGES reads the same as a string or an array.
    """
    text = script_text()
    names = READONLY.findall(header(text))
    assert names, "the header holds no readonly NAME=value statement"
    body = HEADER_EMIT + "".join(f"__probe_emit {name}\n" for name in names)
    result = run_bash(tmp_path, probe_prelude(tmp_path) + header(text) + STUB_FUNCTIONS + body)
    assert result.returncode == 0, result.stderr
    assert stub_calls(tmp_path) == [], "the header ran a command"
    fields = result.stdout.split("\0")
    return {fields[i]: (fields[i + 1], fields[i + 2]) for i in range(0, len(fields) - 2, 3)}


def package_table(values: dict[str, tuple[str, str]]) -> dict[str, tuple[str, int, str]]:
    """Return the PACKAGES entries as name -> (version, size, sha256); fail on a malformed or repeated entry."""
    assert "PACKAGES" in values, "the header has no readonly PACKAGES"
    table: dict[str, tuple[str, int, str]] = {}
    for line in values["PACKAGES"][1].splitlines():
        if not line.strip():
            continue
        parts = line.split()
        assert len(parts) == 3, f"not '<name>=<version> <size> <sha256>': {line!r}"
        name, equals, version = parts[0].partition("=")
        assert equals and version, f"no =<version>: {line!r}"
        assert parts[1].isdigit() and HEX64.fullmatch(parts[2]), f"bad size or sha256: {line!r}"
        assert name not in table, f"{name} is listed twice"
        table[name] = (version, int(parts[1]), parts[2])
    return table


def triple(values: dict[str, tuple[str, str]], stem: str) -> tuple[str, int, str]:
    """Return (URL, size, sha256) of one download in the header."""
    return values[f"{stem}_URL"][1], int(values[f"{stem}_SIZE"][1]), values[f"{stem}_SHA256"][1]


def sha(data: bytes) -> str:
    """Return the sha256 hex digest of `data`."""
    return hashlib.sha256(data).hexdigest()


# The script as text.


def test_script_is_strict_bash_in_plain_ascii_with_lf() -> None:
    """A bash script with strict options, every byte ASCII, no carriage return."""
    assert SCRIPT.is_file(), f"{SCRIPT_REL} does not exist yet"
    data = SCRIPT.read_bytes()
    assert all(byte < 128 for byte in data), "a non-ASCII byte"
    assert b"\r" not in data, "a CR line end"
    text = data.decode("ascii")
    assert text.startswith("#!/usr/bin/env bash\n")
    assert "set -euo pipefail" in header(text)


@needs_bash
def test_script_parses_as_bash() -> None:
    """bash -n accepts the script."""
    script_text()
    assert BASH is not None
    # A relative POSIX path, since bash on Windows would read backslashes in a native path as escapes.
    result = subprocess.run([BASH, "-n", SCRIPT_REL], capture_output=True, text=True, cwd=REPO, timeout=60)
    assert result.returncode == 0, result.stderr


def test_header_holds_only_comments_set_and_readonly_statements() -> None:
    """Every pinned value sits in one readonly block at the top, with nothing that runs a program."""
    open_end: str | None = None
    for line in code_lines(header(script_text())):
        if open_end is not None:
            if line.endswith(open_end):
                open_end = None
            continue
        if line.startswith("set -"):
            continue
        match = re.match(r"readonly (?:-a )?[A-Z][A-Z0-9_]*=(.*)$", line)
        assert match, f"the header runs or sets something other than a readonly value: {line!r}"
        value = match.group(1)
        if value.startswith("(") and not value.endswith(")"):
            open_end = ")"
        elif value.startswith('"') and value.count('"') == 1:
            open_end = '"'
    assert open_end is None, "a multi-line readonly value is not closed"


def test_header_names_its_sources_and_the_date_read() -> None:
    """The comments of the readonly block name the indexes and the date they were read."""
    comments = "\n".join(line for line in header(script_text()).splitlines() if line.lstrip().startswith("#"))
    assert "2026-10-17" in comments
    assert "apt.llvm.org" in comments
    assert "archive.ubuntu.com" in comments or "packages.ubuntu.com" in comments


@needs_bash
def test_every_download_has_a_url_a_size_and_a_sha256(tmp_path: Path) -> None:
    """Each NAME_URL has NAME_SIZE in bytes and a 64-hex NAME_SHA256, all readonly; the four downloads are there."""
    values = header_values(tmp_path)
    for name, (declared, _value) in values.items():
        assert "r" in declared.split()[1], f"{name} is not readonly"
    stems = sorted(name[: -len("_URL")] for name in values if name.endswith("_URL"))
    assert {"ROOTFS", "LLVM_KEY", "CMAKE", "UV"} <= set(stems), stems
    for stem in stems:
        assert values[f"{stem}_URL"][1].startswith("https://"), stem
        assert f"{stem}_SIZE" in values and values[f"{stem}_SIZE"][1].isdigit(), f"{stem} has no size"
        assert f"{stem}_SHA256" in values and HEX64.fullmatch(values[f"{stem}_SHA256"][1]), f"{stem} has no sha256"


@needs_bash
def test_downloads_are_the_published_files(tmp_path: Path) -> None:
    """The image, the LLVM key, CMake, and uv carry the URL, size, and sha256 their pages published on 2026-10-17."""
    values = header_values(tmp_path)
    url, size, digest = triple(values, "ROOTFS")
    assert url in ROOTFS_CHOICES, url
    assert (size, digest) == ROOTFS_CHOICES[url]
    assert triple(values, "LLVM_KEY") == LLVM_KEY
    if "LLVM_KEY_FINGERPRINT" in values:
        fingerprint = values["LLVM_KEY_FINGERPRINT"][1].upper()
        assert HEX40.fullmatch(fingerprint) and fingerprint == LLVM_KEY_FINGERPRINT, fingerprint
    assert triple(values, "CMAKE") == CMAKE
    assert triple(values, "UV") == UV


@needs_bash
def test_distro_name_is_the_constant(tmp_path: Path) -> None:
    """DISTRO is LASSI-Ubuntu-22.04, readonly."""
    values = header_values(tmp_path)
    assert values.get("DISTRO", ("", ""))[1] == DISTRO


@needs_bash
def test_packages_are_the_named_set_at_the_index_versions(tmp_path: Path) -> None:
    """PACKAGES lists exactly the named packages, each with the version, size, and sha256 its index served."""
    table = package_table(header_values(tmp_path))
    assert sorted(table) == sorted(PACKAGES)
    for name, pinned in PACKAGES.items():
        assert table[name] == pinned, name


def apt_install_words(line: str) -> list[str] | None:
    """Return the literal package words of an apt-get or apt install command line, or None if it has none."""
    words = tokens(line)
    for start, word in enumerate(words):
        if word.rsplit("/", 1)[-1] not in ("apt-get", "apt") or "install" not in words[start + 1 :]:
            continue
        rest = words[words.index("install", start + 1) + 1 :]
        found: list[str] = []
        skip = False
        for index, item in enumerate(rest):
            following = rest[index + 1] if index + 1 < len(rest) else ""
            if set(item) <= PUNCTUATION or (item.isdigit() and following[:1] in ("<", ">") and following):
                break
            if skip:
                skip = False
            elif item in ("-o", "--option", "-t", "--target-release"):
                skip = True
            elif not item.startswith("-") and "$" not in item:
                found.append(item)
        return found
    return None


def test_every_apt_package_the_script_names_carries_its_version() -> None:
    """No apt install command names a package without =<version>; the .debs are downloaded first, then kept."""
    text = script_text()
    installs = [(line, words) for line in code_lines(text) if (words := apt_install_words(line)) is not None]
    assert installs, "the script runs no apt-get install"
    for line, words in installs:
        for word in words:
            assert re.fullmatch(r"[a-z0-9][a-z0-9.+-]*=\S+", word), f"{word!r} has no =<version>: {line}"
    assert any("--download-only" in tokens(line) for line, _words in installs), "no download-only step"
    assert "Dir::Cache::archives" in text
    assert "install/batch1/debs" in text


def test_apt_never_removes_a_package() -> None:
    """Every apt-get or apt install command carries --no-remove, so resolving the named set removes nothing."""
    installs = [line for line in code_lines(script_text()) if apt_install_words(line) is not None]
    assert installs, "the script runs no apt-get install"
    for line in installs:
        assert "--no-remove" in tokens(line), line


# The versions the pin files expect.


@needs_bash
def test_clang_20_and_lld_20_are_the_build_tt_metal_pin_names(tmp_path: Path) -> None:
    """clang-20's version is the build in tt-metal.pin's EXPECT_VERSION; lld-20 is the same build."""
    table = package_table(header_values(tmp_path))
    pin = read_pin("tt-metal")
    match = re.fullmatch(r"1:(\d+\.\d+\.\d+)~(\+\+\S+)", table["clang-20"][0])
    assert match, table["clang-20"][0]
    assert f"Ubuntu clang version {match.group(1)} ({match.group(2)})" == pin["EXPECT_VERSION"]
    assert pin["CLANG_EXPECT"] == f"clang version {match.group(1)}"
    assert table["lld-20"][0] == table["clang-20"][0]
    assert pin["LINKER_EXPECT"] == f"LLD {match.group(1)}"


@needs_bash
def test_clang_17_and_lld_17_are_the_build_tt_mlir_pin_names(tmp_path: Path) -> None:
    """clang-17 and lld-17 carry the versions tt-mlir.pin expects, at the build section 2.3 read."""
    table = package_table(header_values(tmp_path))
    pin = read_pin("tt-mlir")
    match = re.fullmatch(r"1:(\d+\.\d+\.\d+)~\+\+\S+", table["clang-17"][0])
    assert match, table["clang-17"][0]
    assert pin["CLANG_EXPECT"] == f"clang version {match.group(1)}"
    assert pin["LINKER_EXPECT"] == f"LLD {match.group(1)}"
    assert table["lld-17"][0] == table["clang-17"][0] == LLVM17


@needs_bash
def test_g_plus_plus_12_is_the_build_gcc_pin_names(tmp_path: Path) -> None:
    """g++-12's package version is the one gcc.pin's EXPECT_VERSION banner names."""
    table = package_table(header_values(tmp_path))
    pin = read_pin("gcc")
    assert f"(Ubuntu {table['g++-12'][0]}) {pin['VERSION']}" == pin["EXPECT_VERSION"]


@needs_bash
def test_python_and_ninja_are_the_versions_the_pins_expect(tmp_path: Path) -> None:
    """python3.10 and its venv and dev packages are one 3.10.12 build; ninja-build is 1.10.1, as both pins expect."""
    table = package_table(header_values(tmp_path))
    mlir, metal = read_pin("tt-mlir"), read_pin("tt-metal")
    python = table["python3.10"][0]
    assert mlir["PYTHON_EXPECT"] == f"Python {python.split('-')[0]}"
    assert table["python3.10-venv"][0] == table["python3.10-dev"][0] == python
    ninja = table["ninja-build"][0].split("-")[0]
    assert ninja == mlir["NINJA_EXPECT"] == metal["NINJA_EXPECT"]


@needs_bash
def test_cmake_and_uv_are_the_pinned_releases(tmp_path: Path) -> None:
    """CMake is the version both pins expect at /usr/local/bin; uv is the version the repository's CI installs."""
    values = header_values(tmp_path)
    assert "CMAKE_VERSION" in values and "UV_VERSION" in values
    version = values["CMAKE_VERSION"][1]
    for name in ("tt-metal", "tt-mlir"):
        pin = read_pin(name)
        assert pin["CMAKE_EXPECT"] == f"cmake version {version}", name
        assert pin["CMAKE"] == "/usr/local/bin/cmake", name
    assert values["CMAKE_URL"][1].endswith(f"/v{version}/cmake-{version}-linux-x86_64.tar.gz")
    ci = re.search(r"\buv==(\S+)", CPU_TESTS.read_text(encoding="utf-8"))
    assert ci, "the CI workflow names no uv version"
    assert values["UV_VERSION"][1] == ci.group(1)
    assert f"/{ci.group(1)}/uv-x86_64-unknown-linux-gnu.tar.gz" in values["UV_URL"][1]


def test_version_strings_are_read_from_the_pin_files_at_run_time() -> None:
    """The final check reads the three pin files; no code line copies the banners gcc.pin and tt-metal.pin hold."""
    code = "\n".join(code_lines(script_text()))
    assert ".pin" in code, "no code line reads a pin file"
    for name in ("gcc", "tt-metal", "tt-mlir"):
        assert re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", code), f"no code line names {name}.pin"
    for name in ("gcc", "tt-metal"):
        assert read_pin(name)["EXPECT_VERSION"] not in code, f"{name}.pin's EXPECT_VERSION is copied, not read"


# What the stages do, as text.


def test_import_stage_imports_the_named_distro_as_wsl_2() -> None:
    """The import is `wsl.exe --import` with --version 2, and the owner is told to terminate the distro after."""
    text = script_text()
    imports = [tokens(line) for line in code_lines(text) if "--import" in tokens(line)]
    assert imports, "no wsl.exe --import line"
    for words in imports:
        assert "wsl.exe" in words and "--version" in words, words
        assert words[words.index("--version") + 1 :][:1] == ["2"], words
    assert "--terminate" in text
    assert "system32" not in text.lower(), "wsl.exe is called by name, never by a Windows path"


def test_packages_stage_writes_wsl_conf_and_its_log() -> None:
    """/etc/wsl.conf gets systemd and the default user; the stage logs through tee and records dpkg-query -W."""
    text = script_text()
    for needle in ("/etc/wsl.conf", "[boot]", "systemd=true", "[user]", "default="):
        assert needle in text, needle
    assert "install/batch1/install.log" in text
    lines = code_lines(text)
    assert any("tee" in tokens(line) for line in lines), "no tee"
    assert any("dpkg-query" in tokens(line) and "-W" in tokens(line) for line in lines), "no dpkg-query -W"
    assert any("tar" in tokens(line) and "/usr/local" in line for line in lines), "CMake is not unpacked in /usr/local"


def test_user_is_created_plainly_with_no_sudo_and_no_password() -> None:
    """useradd -m -s /bin/bash, no group option and no password option; no sudo command, group, or rule anywhere."""
    lines = code_lines(script_text())
    adds = [tokens(line) for line in lines if "useradd" in tokens(line)]
    assert adds, "no useradd"
    for words in adds:
        assert "-m" in words and "-s" in words and words[words.index("-s") + 1] == "/bin/bash", words
        assert not {"-G", "--groups", "-p", "--password", "-aG"} & set(words), words
    for line in lines:
        assert "sudoers" not in line, line
        words = tokens(line)
        for index, word in enumerate(words):
            if in_command_position(words, index):
                assert word not in ("sudo", "usermod", "adduser", "passwd", "chpasswd"), line


def test_script_never_names_alpha01s_root_or_a_refused_directory_outside_its_check() -> None:
    """No /mnt/nvme10 anywhere; /tmp, /var/tmp, and /opt appear only in check_scratch_root."""
    text = script_text()
    assert "/mnt/nvme10" not in text and "joseph_ufl" not in text
    rest = text.replace(functions(text).get("check_scratch_root", ""), "")
    for line in code_lines(rest):
        assert not re.search(r'(^|[\s"=(:|])/(var/tmp|tmp|opt)(/|\b)', line), line


# Argument handling, run in bash with stand-ins.


def quoted(args: list[str]) -> str:
    """Return `set -- <args>` with each argument single-quoted for bash."""
    return "set -- " + " ".join(shlex.quote(arg) for arg in args) + "\n"


def assert_refused(result: subprocess.CompletedProcess[str], *names: str) -> None:
    """The probe stopped before its end, nonzero, with an install-batch1 message naming each of `names`."""
    assert result.returncode != 0, result.stdout + result.stderr
    assert "PROBE-DONE" not in result.stdout
    assert PREFIX in result.stderr, result.stderr
    for name in names:
        assert name in result.stderr, (name, result.stderr)


def assert_passed(result: subprocess.CompletedProcess[str]) -> None:
    """The probe ran to its end."""
    assert result.returncode == 0 and "PROBE-DONE" in result.stdout, result.stdout + result.stderr


IMPORT_REFUSED = [
    ([], ["--disk-dir"]),
    (["--download-dir", "C:/Downloads"], ["--disk-dir"]),
    (["--disk-dir", "C:/WSL/LASSI"], ["--download-dir"]),
    (["--disk-dir", "", "--download-dir", "C:/Downloads"], ["--disk-dir"]),
    (["--disk-dir", "C:/WSL/LASSI", "--download-dir", ""], ["--download-dir"]),
    (["--download-dir", "C:/Downloads", "--disk-dir"], ["--disk-dir"]),
    (["--disk-dir", "C:/WSL/LASSI", "--download-dir", "C:/Downloads", "--force"], ["--force"]),
]


@needs_bash
@pytest.mark.parametrize(("args", "named"), IMPORT_REFUSED, ids=repr)
def test_parse_import_args_refuses_a_missing_or_unknown_argument(
    tmp_path: Path, args: list[str], named: list[str]
) -> None:
    """Both import arguments are required, with no default; an unknown one is refused by name."""
    result = run_functions(tmp_path, quoted(args) + 'parse_import_args "$@"\necho PROBE-DONE\n')
    assert_refused(result, *named)
    assert stub_calls(tmp_path) == []


@needs_bash
@pytest.mark.parametrize(
    "args",
    [
        ["--disk-dir", "C:\\WSL\\LASSI", "--download-dir", "C:\\Users\\J\\Downloads"],
        ["--download-dir", "Z:/dl", "--disk-dir", "Z:/wsl/lassi"],
    ],
    ids=["in-order", "reversed"],
)
def test_parse_import_args_sets_both_values(tmp_path: Path, args: list[str]) -> None:
    """Given both, the parser sets disk_dir and download_dir as given and runs nothing."""
    body = quoted(args) + 'parse_import_args "$@"\nprintf "%s\\n" "$disk_dir" "$download_dir"\necho PROBE-DONE\n'
    result = run_functions(tmp_path, body)
    assert result.returncode == 0, result.stderr
    given = dict(zip(args[::2], args[1::2], strict=True))
    assert result.stdout.splitlines() == [given["--disk-dir"], given["--download-dir"], "PROBE-DONE"]
    assert stub_calls(tmp_path) == []


DISK_ACCEPTED = ["C:\\WSL\\LASSI", "C:/WSL/LASSI", "c:/wsl", "/c/WSL/LASSI", "Z:\\wsl\\lassi", "z:/wsl", "/z/wsl"]
DISK_REFUSED = ["D:\\WSL\\LASSI", "E:/wsl", "/d/wsl", "/e/wsl", "WSL\\LASSI", "wsl/lassi", "\\\\server\\share\\wsl"]


@needs_bash
@pytest.mark.parametrize("disk_dir", DISK_ACCEPTED)
def test_check_disk_dir_accepts_c_and_z(tmp_path: Path, disk_dir: str) -> None:
    """A disk directory on C: or Z:, in any of Git Bash's spellings, passes."""
    assert_passed(run_functions(tmp_path, quoted([disk_dir]) + 'check_disk_dir "$1"\necho PROBE-DONE\n'))
    assert stub_calls(tmp_path) == []


@needs_bash
@pytest.mark.parametrize("disk_dir", DISK_REFUSED)
def test_check_disk_dir_refuses_other_drives_and_relative_paths(tmp_path: Path, disk_dir: str) -> None:
    """A disk directory on another drive, a relative one, or a network share is refused."""
    assert_refused(run_functions(tmp_path, quoted([disk_dir]) + 'check_disk_dir "$1"\necho PROBE-DONE\n'))
    assert stub_calls(tmp_path) == []


def wsl_lists(cwd: Path, names: list[str]) -> None:
    """Write the wsl.exe stand-in's two lists: UTF-16LE with CRLF, as wsl.exe prints, and UTF-8 for WSL_UTF8=1."""
    text = "".join(f"{name}\r\n" for name in names)
    lists = cwd / "lists"
    lists.mkdir(exist_ok=True)
    (lists / "utf16.txt").write_bytes(text.encode("utf-16-le"))
    (lists / "utf8.txt").write_bytes(text.encode("utf-8"))


@needs_bash
@pytest.mark.parametrize("utf8", [False, True], ids=["utf16", "utf8"])
@pytest.mark.parametrize(
    ("names", "exists"),
    [
        (["Ubuntu-24.04", "docker-desktop"], False),
        (["Ubuntu-24.04", DISTRO, "docker-desktop"], True),
        ([DISTRO], True),
        ([f"{DISTRO}-old", "Ubuntu-24.04"], False),
    ],
    ids=["absent", "present", "only", "longer-name"],
)
def test_check_no_distro_refuses_an_existing_distro(
    tmp_path: Path, names: list[str], exists: bool, utf8: bool
) -> None:
    """The distro list is read with `wsl.exe -l -q`, and only an exact LASSI-Ubuntu-22.04 line stops the import."""
    wsl_lists(tmp_path, names)
    setting = "export WSL_UTF8=1\n" if utf8 else "unset WSL_UTF8\n"
    result = run_functions(tmp_path, setting + "check_no_distro\necho PROBE-DONE\n")
    if exists:
        assert_refused(result, DISTRO)
    else:
        assert_passed(result)
    calls = stub_calls(tmp_path)
    assert calls, "check_no_distro never listed the distros"
    assert all(call.partition(" ")[2] in ("-l -q", "-q -l", "--list --quiet") for call in calls), calls


PACKAGES_REFUSED = [
    ([], ["--user"]),
    (["--root", "/srv/lassi"], ["--user"]),
    (["--user", "lassi"], ["--root"]),
    (["--user", "", "--root", "/srv/lassi"], ["--user"]),
    (["--user", "lassi", "--root", ""], ["--root"]),
    (["--user", "lassi", "--root"], ["--root"]),
    (["--user", "lassi", "--root", "/srv/lassi", "--yes"], ["--yes"]),
]


@needs_bash
@pytest.mark.parametrize(("args", "named"), PACKAGES_REFUSED, ids=repr)
def test_parse_packages_args_refuses_a_missing_or_unknown_argument(
    tmp_path: Path, args: list[str], named: list[str]
) -> None:
    """Both packages arguments are required, with no default; an unknown one is refused by name."""
    result = run_functions(tmp_path, quoted(args) + 'parse_packages_args "$@"\necho PROBE-DONE\n', {"FAKE_UID": "0"})
    assert_refused(result, *named)
    assert stub_calls(tmp_path) == []


@needs_bash
def test_parse_packages_args_sets_both_values(tmp_path: Path) -> None:
    """Given both, the parser sets user_name and scratch_root as given, with no check and no file written."""
    body = quoted(["--root", "/srv/lassi", "--user", "lassi"])
    body += 'parse_packages_args "$@"\nprintf "%s\\n" "$user_name" "$scratch_root"\necho PROBE-DONE\n'
    result = run_functions(tmp_path, body, {"FAKE_UID": "0"})
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["lassi", "/srv/lassi", "PROBE-DONE"]
    assert stub_calls(tmp_path) == []


ROOT_REFUSED = [
    "relative/root", "./lassi", "/", "/mnt", "/mnt/c/lassi", "/mnt/z/lassi", "/mnt/wsl/lassi", "/tmp", "/tmp/lassi",
    "/var/tmp", "/var/tmp/lassi", "/opt", "/opt/lassi", "/srv/../mnt/c/lassi", "/srv/../tmp/lassi",
]
ROOT_ACCEPTED = ["/srv/lassi", "/home/lassi/scratch", "/lassi", "/mnt2/lassi", "/tmpdata/lassi"]


@needs_bash
@pytest.mark.parametrize("root", ROOT_REFUSED)
def test_check_scratch_root_refuses_unsafe_roots(tmp_path: Path, root: str) -> None:
    """A relative root, /, or one under /mnt (the Windows drives), /tmp, /var/tmp, or /opt is refused."""
    assert_refused(run_functions(tmp_path, quoted([root]) + 'check_scratch_root "$1"\necho PROBE-DONE\n'))
    assert stub_calls(tmp_path) == []


@needs_bash
@pytest.mark.parametrize("root", ROOT_ACCEPTED)
def test_check_scratch_root_accepts_a_root_on_the_distro_disk(tmp_path: Path, root: str) -> None:
    """An absolute root outside the refused directories passes, names that only share their first letters included."""
    assert_passed(run_functions(tmp_path, quoted([root]) + 'check_scratch_root "$1"\necho PROBE-DONE\n'))
    assert stub_calls(tmp_path) == []


@needs_bash
@pytest.mark.parametrize(("uid", "allowed"), [("0", True), ("1000", False)], ids=["root", "user"])
def test_check_euid_needs_root(tmp_path: Path, uid: str, allowed: bool) -> None:
    """The packages stage runs only as root, read from `id -u`, and never reaches for sudo."""
    result = run_functions(tmp_path, "check_euid\necho PROBE-DONE\n", {"FAKE_UID": uid})
    if allowed:
        assert_passed(result)
    else:
        assert_refused(result, "root")
    assert stub_calls(tmp_path) == []


DEB = b"SYNTHETIC .deb bytes for the check_file probe; not a package.\n"


@needs_bash
@pytest.mark.parametrize(
    ("name", "size", "digest", "good"),
    [
        ("pkg.deb", len(DEB), sha(DEB), True),
        ("pkg.deb", len(DEB) + 1, sha(DEB), False),
        ("pkg.deb", len(DEB), sha(DEB + b"x"), False),
        ("missing.deb", len(DEB), sha(DEB), False),
    ],
    ids=["matches", "wrong-size", "wrong-sha256", "missing"],
)
def test_check_file_checks_size_and_sha256(tmp_path: Path, name: str, size: int, digest: str, good: bool) -> None:
    """A fetched file is used only when its size and sha256 both match the pinned values."""
    (tmp_path / "pkg.deb").write_bytes(DEB)
    result = run_functions(tmp_path, quoted([name, str(size), digest]) + 'check_file "$1" "$2" "$3"\necho PROBE-DONE\n')
    if good:
        assert_passed(result)
    else:
        assert_refused(result, name)
    assert stub_calls(tmp_path) == []


# The packages stage's order and its clean-up, run in bash with stand-ins.


STAGE_ROOT = "/home/lassi/scratch"
# run_packages's steps, each a stand-in that logs its name, in the order the stage runs them; own_root, which logs
# what ensure_user did and gives the root to the user, runs for real between record_host start and write_wsl_conf.
STEPS_BEFORE_OWN_ROOT = ("record_host start",)
STEPS_AFTER_OWN_ROOT = (
    "write_wsl_conf", "add_llvm_repo", "install_apt_packages", "install_cmake", "install_uv", "give_install_tree",
    "check_versions", "record_host end",
)
# The steps that write under the root; the install tree goes to the user after all of them.
WRITING_STEPS = ("write_wsl_conf", "add_llvm_repo", "install_apt_packages", "install_cmake", "install_uv")
# A new user, lassi: `id` knows it only once the useradd stand-in has run; uid 0 for `id -u` alone (check_euid).
NEW_USER_STAND_INS = """useradd() { printf '%s\\n' "useradd $*" >> "$STUB_LOG"; : > "$PWD/user-made"; }
id() {
  case "$*" in
    -u) printf '0\\n' ;;
    '-u lassi') if [ -f "$PWD/user-made" ]; then printf '1000\\n'; else return 1; fi ;;
    '-nG lassi' | '-gn lassi') printf 'lassi\\n' ;;
    lassi) printf 'uid=1000(lassi) gid=1000(lassi) groups=1000(lassi)\\n' ;;
    *) return 1 ;;
  esac
}
"""


def run_stage_with_steps_stubbed(cwd: Path) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    """Run packages_stage for lassi at STAGE_ROOT with its host check, mkdir, tee, and run_packages's steps stubbed.

    Nothing touches the file system or the host: mkdir, install, and chown only log, and tee passes its input on.
    """
    steps = {step.split()[0] for step in STEPS_BEFORE_OWN_ROOT + STEPS_AFTER_OWN_ROOT}
    body = (
        'script_path="$PWD/probe.sh"\n'
        + "".join(log_stub(name) for name in ("check_packages_host", "mkdir", "install", *sorted(steps)))
        + 'tee() { printf "%s\\n" "tee $*" >> "$STUB_LOG"; cat; }\n'
        + NEW_USER_STAND_INS
        + f"packages_stage --user lassi --root {STAGE_ROOT}\necho PROBE-DONE\n"
    )
    result = run_functions(cwd, body)
    assert_passed(result)
    return result, [call.strip() for call in stub_calls(cwd) if not call.startswith("tee ")]


@needs_bash
def test_packages_stage_creates_the_user_before_any_directory_under_the_root(tmp_path: Path) -> None:
    """useradd -m runs before the first mkdir under the root, so a root in the user's home leaves that home its own.

    Its result reaches the log, through tee, once the log exists.
    """
    result, calls = run_stage_with_steps_stubbed(tmp_path)
    adds = [index for index, call in enumerate(calls) if call.startswith("useradd ")]
    assert adds, f"useradd never ran: {calls}"
    assert calls[adds[0]] == "useradd -m -s /bin/bash lassi", calls
    under_root = [
        index for index, call in enumerate(calls)
        if call.split()[0] in ("mkdir", "install") and STAGE_ROOT in call
    ]
    assert under_root, f"no directory was made under the root: {calls}"
    assert calls.index("check_packages_host") < adds[0] < under_root[0], calls
    assert any(f"{STAGE_ROOT}/install/batch1" in calls[index] for index in under_root), calls
    assert "created user lassi" in result.stdout, result.stdout


@needs_bash
def test_run_packages_gives_the_install_tree_to_the_user_after_every_write(tmp_path: Path) -> None:
    """own_root gives the root to the user before the first write; give_install_tree runs after the last one."""
    _result, calls = run_stage_with_steps_stubbed(tmp_path)
    names = {step.split()[0] for step in STEPS_BEFORE_OWN_ROOT + STEPS_AFTER_OWN_ROOT} | {"chown"}
    steps = [call for call in calls if call.split()[0] in names]
    own = f"chown lassi:lassi {STAGE_ROOT}"
    assert steps == [*STEPS_BEFORE_OWN_ROOT, own, *STEPS_AFTER_OWN_ROOT], calls
    assert all(steps.index(own) < steps.index(step) < steps.index("give_install_tree") for step in WRITING_STEPS)


@needs_bash
def test_give_install_tree_chowns_the_tree_and_reads_du_back_as_the_user(tmp_path: Path) -> None:
    """<root>/install, kept .debs and log included, goes to the user, and du -sk of the root then runs as the user."""
    body = f'root={STAGE_ROOT}\nuser_name=lassi\nuser_group=lassi\ngive_install_tree\necho PROBE-DONE\n'
    assert_passed(run_functions(tmp_path, body))
    calls = stub_calls(tmp_path)
    assert len(calls) == 2, calls
    chown, runuser = (call.split() for call in calls)
    assert chown[0] == "chown" and "-R" in chown and chown[-2:] == ["lassi:lassi", f"{STAGE_ROOT}/install"], calls
    assert not {"-L", "-H", "--dereference"} & set(chown), calls
    assert runuser[:3] == ["runuser", "-u", "lassi"] and runuser[-3:] == ["du", "-sk", STAGE_ROOT], calls


GNUPG_STAND_INS = """fetch() { printf '%s\\n' "fetch $*" >> "$STUB_LOG"; }
gpg() {
  printf '%s\\n' "gpg GNUPGHOME=${GNUPGHOME:-} $*" >> "$STUB_LOG"
  [ -d "${GNUPGHOME:-}" ] || { printf '%s\\n' "gpg stand-in: no GnuPG home" >&2; return 1; }
  case " $* " in
    *" --show-keys "*) printf 'pub:-:4096:1:15CF4D18AF4F7421:::::::::\\nfpr:::::::::%s:\\n' "$LLVM_KEY_FINGERPRINT" ;;
  esac
}
rm() { printf '%s\\n' "rm $*" >> "$STUB_LOG"; command rm "$@"; }
"""


@needs_bash
def test_install_llvm_key_removes_its_gnupg_home_once_the_key_is_dearmored(tmp_path: Path) -> None:
    """The key is checked and dearmored in a GnuPG home made for it under the root, which is then removed, alone."""
    batch = tmp_path / "root" / "install" / "batch1"
    batch.mkdir(parents=True)
    (batch / "llvm-snapshot.gpg.key").write_bytes(b"SYNTHETIC key bytes for the probe; not a key.\n")
    body = (
        'root="$PWD/root"\n' + log_stub("install") + log_stub("chmod") + GNUPG_STAND_INS
        + "install_llvm_key /etc/apt/keyrings/lassi-llvm-snapshot.gpg\necho PROBE-DONE\n"
    )
    assert_passed(run_functions(tmp_path, body))
    calls = stub_calls(tmp_path)
    gpgs = [call for call in calls if call.startswith("gpg ")]
    assert len(gpgs) == 2 and "--show-keys" in gpgs[0] and "--dearmor" in gpgs[1], calls
    home = gpgs[0].split()[1].partition("=")[2]
    assert re.search(r"/root/install/batch1/gnupg\.[A-Za-z0-9]{6}$", home), home
    assert gpgs[1].split()[1] == f"GNUPGHOME={home}", calls
    removals = [call for call in calls if call.startswith("rm ")]
    assert removals and removals[-1].split()[-1] == home, calls
    assert calls.index(removals[-1]) > calls.index(gpgs[1]), "the GnuPG home went before the key was dearmored"
    assert sorted(path.name for path in batch.iterdir()) == ["llvm-snapshot.gpg.key"]


GNUPG_REFUSED = [
    ('"$PWD/root"', '"$root"'),
    ('"$PWD/root"', '"$root/install/batch1"'),
    ('"$PWD/root"', '"$root/install/batch1/gnupg"'),
    ('"$PWD/root"', '"$root/install/batch1/gnupg.ab/../"'),
    ('"$PWD/root"', '"$root/install/batch1/gnupg.abcdefg"'),
    ('"$PWD/root"', '"$root/install/batch1/sub/gnupg.abcdef"'),
    ('"$PWD/root"', '"$root/install/gnupg.abcdef"'),
    ('"$PWD/root"', '"/install/batch1/gnupg.abcdef"'),
    ('"$PWD/root"', '""'),
    ('""', '"/install/batch1/gnupg.abcdef"'),
    ("relative", '"relative/install/batch1/gnupg.abcdef"'),
]
GNUPG_TREE = ("install/batch1/gnupg.abcdef/x", "install/batch1/gnupg", "install/batch1/sub/gnupg.abcdef/x",
              "install/gnupg.abcdef/x")


def gnupg_tree(cwd: Path) -> list[Path]:
    """Make the directories the remove_gnupg_home probes aim at, under cwd/root, and return them."""
    made = [cwd / "root" / leaf for leaf in GNUPG_TREE]
    for path in made:
        path.mkdir(parents=True)
    return made


@needs_bash
@pytest.mark.parametrize(("root", "target"), GNUPG_REFUSED, ids=repr)
def test_remove_gnupg_home_refuses_any_other_path(tmp_path: Path, root: str, target: str) -> None:
    """Only a gnupg.<six letters or digits> directory directly in <root>/install/batch1, root absolute, is removed."""
    made = gnupg_tree(tmp_path)
    body = log_stub("rm") + f"root={root}\nremove_gnupg_home {target}\necho PROBE-DONE\n"
    assert_refused(run_functions(tmp_path, body))
    assert stub_calls(tmp_path) == []
    assert all(path.is_dir() for path in made)


@needs_bash
def test_remove_gnupg_home_removes_the_one_directory(tmp_path: Path) -> None:
    """The GnuPG home itself goes, with what gpg left in it; its neighbours stay."""
    made = gnupg_tree(tmp_path)
    body = (
        'rm() { printf "%s\\n" "rm $*" >> "$STUB_LOG"; command rm "$@"; }\nroot="$PWD/root"\n'
        'remove_gnupg_home "$root/install/batch1/gnupg.abcdef"\necho PROBE-DONE\n'
    )
    assert_passed(run_functions(tmp_path, body))
    assert [call.split()[:2] for call in stub_calls(tmp_path)] == [["rm", "-rf"]]
    assert not made[0].parent.exists()
    assert all(path.is_dir() for path in made[1:])


# The whole script, only where it must stop before it runs anything.


WHOLE = """set -euo pipefail
chmod +x stubs/*
export STUB_LOG="$PWD/stub.log"
: > "$STUB_LOG"
script="$1"
shift
if command -v cygpath >/dev/null 2>&1; then script="$(cygpath -u "$script")"; fi
export PATH="$PWD/stubs:/usr/bin:/bin"
exec bash "$script" "$@"
"""


@needs_bash
@pytest.mark.parametrize(
    ("args", "named"),
    [([], ["import", "packages"]), (["deploy"], ["deploy"]), (["import"], ["--disk-dir"]), (["packages"], [])],
    ids=["no-stage", "unknown-stage", "import-bare", "packages-bare"],
)
def test_script_refuses_before_running_anything(tmp_path: Path, args: list[str], named: list[str]) -> None:
    """With no stage, an unknown stage, or a stage without its arguments, the script stops with a message at once."""
    script_text()
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    for name in PLAIN_STUBS:
        (stubs / name).write_bytes(STUB_SCRIPT.replace("{name}", name).encode("ascii"))
    write_probe(tmp_path, WHOLE)
    assert BASH is not None
    result = subprocess.run(
        [BASH, "probe.sh", SCRIPT.as_posix(), *args], cwd=tmp_path, capture_output=True, text=True, timeout=60
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert PREFIX in result.stderr, result.stderr
    assert "command not found" not in result.stderr, result.stderr
    for name in named:
        assert name in result.stderr, (name, result.stderr)
    assert stub_calls(tmp_path) == []
