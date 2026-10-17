#!/usr/bin/env bash
# The workstation's first install batch (task P5.15; OQ-055, decision 2 (b); plans/local-plan.md section 2.3 and
# section 7, decision 2; bible Host Facts, Workstation, Workstation scratch, and Workstation approvals; AGENTS.md,
# Local Execution). The owner reviews this script and runs it; no test runs a stage. The script and its output are
# the install record. Each stage takes two arguments, both required with no default, and refuses before it changes
# anything when one is missing or wrong.
#
# Stage 1, import, from Git Bash on Windows:
#   bash tools/local/install-batch1.sh import --disk-dir <directory on C: or Z:> --download-dir <directory>
# It refuses a disk directory on any other drive, a drive with less free space than the floor plus this batch, and
# a WSL distro named LASSI-Ubuntu-22.04 that already exists. It fetches the pinned Ubuntu 22.04 WSL image into the
# download directory, checks its size and sha256, and imports it with wsl.exe --import as the WSL 2 distro
# LASSI-Ubuntu-22.04, whose disk (ext4.vhdx) then lies in the disk directory. It logs to
# install-batch1-import.log in the download directory.
#
# Stage 2, packages, inside the new distro as root, from PowerShell or cmd (from Git Bash, put MSYS_NO_PATHCONV=1
# in front, or Git Bash rewrites the Linux paths), as one line:
#   wsl.exe -d LASSI-Ubuntu-22.04 -u root --exec bash /mnt/c/dev/lassi/tools/local/install-batch1.sh packages
#     --user <linux user name> --root <scratch root>
# It refuses unless it runs as root in that distro (it never calls sudo), and refuses a scratch root that is
# relative, "/", or off the distro's own ext4 disk. It creates the user (useradd -m -s /bin/bash: no password, no
# sudo group) before any directory under the scratch root exists, so a root inside the user's home leaves that home
# the user's own, with its skeleton files; then makes the scratch root, owned by the user; writes /etc/wsl.conf
# ([boot] systemd=true, [user] default=<user>), keeping a copy of any earlier file under the scratch root; adds
# apt.llvm.org's two jammy suites, their key checked by size, sha256, and fingerprint in a GnuPG home it makes for
# the key and removes once the key is dearmored; downloads every named package at its pinned version, with its
# dependencies, into <root>/install/batch1/debs, where they are kept, checks each named .deb's size and sha256, and
# only then installs from those files, never removing a package (apt-get --no-remove); logs dpkg-query -W; unpacks
# CMake into /usr/local (so /usr/local/bin/cmake, the pinned path); puts uv in <root>/bin; gives <root>/install,
# the kept .debs and the log included, to the user (apt leaves debs/partial to _apt, mode 0700) and runs du -sk of
# the scratch root as the user, so tools/local/longjob.py can read it; and prints each tool's version line beside
# the string its pin file expects (toolchains/gcc.pin, tt-metal.pin, and tt-mlir.pin, read at run time), failing if
# one differs. Once the scratch root is made, all it prints also goes to <root>/install/batch1/install.log, starting
# with the host record and then what it did about the user. Apart from what apt, useradd, /etc/wsl.conf, and /usr/local need, it writes only
# under the scratch root. Then run wsl.exe --terminate LASSI-Ubuntu-22.04, so that the distro starts again with
# systemd.
set -euo pipefail

# The distro's name, a constant chosen in this script.
readonly DISTRO=LASSI-Ubuntu-22.04
# Every pinned value below was read on 2026-10-17 from the index or release page its comment names: index reads,
# not measurements. A file that no longer matches is refused; re-read the page and re-pin.
#
# The Ubuntu 22.04.5 WSL image: the current official Ubuntu 22.04 WSL image, and alpha01's point release (Ubuntu
# 22.04.5, results/p4-tt-install). https://cloud-images.ubuntu.com/wsl/ says WSL image publication has moved; the
# last dated jammy image there is releases/22.04/20240304. sha256 from https://releases.ubuntu.com/22.04.5/SHA256SUMS
# (the same file is served under /jammy/), size from the file's Content-Length, Last-Modified 2026-03-09. The file
# is a gzip tar. Its manifest (ubuntu-22.04.5-wsl-amd64.manifest) lists util-linux 2.37.2-4ubuntu3.4, systemd
# 249.11-0ubuntu3.17, apt 2.4.14, curl, gpg, and wsl-setup 0.5.8~22.04.1, whose first-run script asks for a user
# only "if not created yet" (its changelog); the packages stage creates the user before any interactive start.
readonly ROOTFS_URL=https://releases.ubuntu.com/22.04.5/ubuntu-22.04.5-wsl-amd64.wsl
readonly ROOTFS_SIZE=360684292
readonly ROOTFS_SHA256=4499c4fe257f2fc83145b429ce211a0a43fd590e70d6261ede616210947d9f8f
# apt.llvm.org's signing key: size and sha256 as https://apt.llvm.org/llvm-snapshot.gpg.key served it, and its
# primary key's fingerprint as gpg --show-keys printed it (uid "Sylvestre Ledru - Debian LLVM packages").
readonly LLVM_KEY_URL=https://apt.llvm.org/llvm-snapshot.gpg.key
readonly LLVM_KEY_SIZE=3167
readonly LLVM_KEY_SHA256=8b2a587ffd672c4687e7581dad4b2f6c1bb2ad6b480cd9771ba2ff48e0b8c75d
readonly LLVM_KEY_FINGERPRINT=6084F3CF814B57C1CF12EFD515CF4D18AF4F7421
# apt.llvm.org's jammy repository and the two suites the clang and lld packages come from. apt fetches and checks
# their signed indexes itself; neither is a download of this script's.
readonly LLVM_APT_REPO=https://apt.llvm.org/jammy/
readonly LLVM_APT_SUITES="llvm-toolchain-jammy-17 llvm-toolchain-jammy-20"
# CMake 3.31.6, the version toolchains/tt-metal.pin and tt-mlir.pin expect at /usr/local/bin/cmake, from Kitware's
# release page https://github.com/Kitware/CMake/releases/tag/v3.31.6: sha256 from cmake-3.31.6-SHA-256.txt, size
# from the asset's Content-Length.
readonly CMAKE_VERSION=3.31.6
readonly CMAKE_URL=https://github.com/Kitware/CMake/releases/download/v3.31.6/cmake-3.31.6-linux-x86_64.tar.gz
readonly CMAKE_SIZE=55010959
readonly CMAKE_SHA256=5a1133ff103c71eb5120e2cc3de922733e7d8a26a98ae716397e8676adb367bf
# uv 0.12.17, the version the repository uses on Windows and in CI (.github/workflows/cpu-tests.yml), from its
# release page https://github.com/astral-sh/uv/releases/tag/0.12.17: sha256 from
# uv-x86_64-unknown-linux-gnu.tar.gz.sha256, size from the asset's Content-Length.
readonly UV_VERSION=0.12.17
readonly UV_URL=https://github.com/astral-sh/uv/releases/download/0.12.17/uv-x86_64-unknown-linux-gnu.tar.gz
readonly UV_SIZE=19755224
readonly UV_SHA256=fa82fd8dde8e8eefdecada6aa0889666556cfceb690d06e0c3bca49eb3070a63
# The named packages, '<name>=<version> <.deb size in bytes> <.deb sha256>', each the highest version its index
# served (dependencies come at the versions apt resolves). clang and lld from apt.llvm.org,
# https://apt.llvm.org/jammy/dists/llvm-toolchain-jammy-17/main/binary-amd64/Packages and the same path for
# llvm-toolchain-jammy-20 (both Release files dated 2026-02-23); the rest from archive.ubuntu.com,
# http://archive.ubuntu.com/ubuntu/dists/{jammy,jammy-updates,jammy-security}/{main,universe}/binary-amd64/Packages.gz
# (jammy-updates dated 2026-10-17 17:52 UTC, jammy-security 2026-10-16 21:49 UTC). clang-17, lld-17, clang-20,
# lld-20, g++-12, ninja-build, and python3.10 are the builds plans/local-plan.md section 2.3 read on 2026-10-15 and
# the pin files name (tt-metal.pin EXPECT_VERSION, tt-mlir.pin, gcc.pin); the rest were read on 2026-10-17 only.
# curl is jammy-security's build; jammy-updates still serves 7.81.0-1ubuntu1.29.
readonly -a PACKAGES=(
  "clang-17=1:17.0.6~++20231209124227+6009708b4367-1~exp1~20231209124336.77 144342 8fc71a5739bfe76d590ff282740ac73735dbf7ac0d1c1c7b3da84628f07d9690"
  "lld-17=1:17.0.6~++20231209124227+6009708b4367-1~exp1~20231209124336.77 1367962 1086e1d6d20d52cb1b6da01812a6fdccab80d8b3efb01dcd10bde7a63c2887e9"
  "clang-20=1:20.1.8~++20250708082409+6fb913d3e2ec-1~exp1~20250708202428.132 151152 51bdb50d21a3ef8589afc777f94d7c24331ee4d646e13dcff7b1dbad00113dc4"
  "lld-20=1:20.1.8~++20250708082409+6fb913d3e2ec-1~exp1~20250708202428.132 1562264 ecad131aabee8b1e76affc788e363bf6e58a4e7584b5af68ee1b926960c2fd21"
  "g++-12=12.3.0-1ubuntu1~22.04.3 12185324 c4123ea47a1d422ec43fb0918f670cfeb7bef1f26e6bf372fe15099fd42bbfcd"
  "ninja-build=1.10.1-1 111368 ca8aae8ac7efad41d1f96a7e227653bd62b24e89cc7ddb18cec262f1869877b2"
  "python3.10=3.10.12-1~22.04.18 508494 10b62fd31bd0310a0b6391c04b67a1f6b33d180820b4a6cf7d25b55b993a15d7"
  "python3.10-venv=3.10.12-1~22.04.18 5724 fb7d9e9e10de015da2eab2a6da33fc6672d096b9281ccac0ecf990de4f724407"
  "python3.10-dev=3.10.12-1~22.04.18 507980 7caa163f4d83ae871f985ec4339fb4c96ae7ef8cb3568b8aec9138f86aacbeb7"
  "git=1:2.34.1-1ubuntu1.17 3173622 8794fcf2c4606c445df0db3dc963c8fb852772208bfb12727a12717c03767af7"
  "curl=7.81.0-1ubuntu1.30 194114 5c59f709ee58e2b12ec9c294b62e2e8cce8f096aa267014e34985cf1960a892f"
  "make=4.3-4.1build1 179742 080b79a1a1623a2e6c6eead37d62b15fdf2c3dbfeafe8ecf5e31c54eb09eadcc"
  "pkg-config=0.29.2-1ubuntu3 48218 69da4190d948d0bf7444ead040791b7924bd1a08a10ec13cd7f0bb9baab09c53"
  "libhwloc-dev=2.7.0-2ubuntu1 255576 a097a8e4596eb3d40a1f069ceab00ea280c527bcd9917fe3f5ae87ec5cdc1b88"
  "libnuma-dev=2.0.14-3ubuntu2 35872 d127bf32ebc3459645bd07fb18dd24f5780f405009589383c533f97d2727643b"
  "libtbb-dev=2021.5.0-7ubuntu2 190718 d3052b9b07c59315cc87d18f7a9ed9c3c5d8d7c446104cdcba6bd17ddfb1fbaa"
  "libcapstone-dev=4.0.2-5 794936 c13611a1dd1a184827cb4d9eccd573522fa59f3ea75bd5c22e5431adeb6b2a6d"
  "libssl-dev=3.0.2-0ubuntu1.30 2377254 8a47dfa7f9bd4d54ac9dc9a1db14891a9133841bf44f12c3db086f2d4321bac6"
  "xz-utils=5.2.5-2ubuntu1.1 84826 012652d4f76a08bfc6badf6a37c6da4c4e4a9f78b8c1560951750a74d700405c"
)
# The drive that holds the distro's disk keeps at least DRIVE_FLOOR_GIB free (Host Facts, Workstation scratch), and
# the import stage wants room for this batch on top: BATCH_GIB, PROJECTED, for the 0.36 GB image and about 2 to 4
# GiB of disk growth in the packages stage (not measured).
readonly DRIVE_FLOOR_GIB=100
readonly BATCH_GIB=5

# say <words>: print one line of progress on stdout (and so into the stage's log).
say() {
  printf 'install-batch1: %s\n' "$*"
}

# warn <words>: print one line on stderr.
warn() {
  printf 'install-batch1: %s\n' "$*" >&2
}

# die <words>: print the reason on stderr and stop with status 2.
die() {
  warn "$@"
  exit 2
}

# import_usage: print stage 1's command line.
import_usage() {
  printf '%s' "usage, from Git Bash on Windows: bash tools/local/install-batch1.sh import --disk-dir <directory on" \
    " C: or Z: for the distro's disk> --download-dir <directory for the image>"
}

# packages_usage: print stage 2's command line, with this checkout's script path once the import stage knows it.
packages_usage() {
  printf '%s' "usage, from PowerShell or cmd: wsl.exe -d $DISTRO -u root --exec bash" \
    " ${wsl_script:-/mnt/c/dev/lassi/tools/local/install-batch1.sh} packages --user <linux user name> --root" \
    " <scratch root on the distro's ext4 disk>"
}

# need_value <usage> <option> [<value> ...]: refuse an option given with no value or an empty one.
need_value() {
  if [ "$#" -lt 3 ] || [ -z "$3" ]; then
    die "$2 needs a value; $1"
  fi
}

# parse_import_args <arguments>: set disk_dir and download_dir from --disk-dir and --download-dir, both required
# with no default; refuse a missing, empty, repeated, or unknown argument, naming it. It reads only its arguments.
parse_import_args() {
  local usage
  usage="$(import_usage)"
  disk_dir=""
  download_dir=""
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --disk-dir)
        need_value "$usage" "$@"
        [ -z "$disk_dir" ] || die "--disk-dir is given twice; $usage"
        disk_dir="$2"
        shift 2
        ;;
      --download-dir)
        need_value "$usage" "$@"
        [ -z "$download_dir" ] || die "--download-dir is given twice; $usage"
        download_dir="$2"
        shift 2
        ;;
      *) die "import: unknown argument '$1'; $usage" ;;
    esac
  done
  [ -n "$disk_dir" ] || die "import: --disk-dir is required and has no default; $usage"
  [ -n "$download_dir" ] || die "import: --download-dir is required and has no default; $usage"
}

# parse_packages_args <arguments>: set user_name and scratch_root from --user and --root, both required with no
# default; refuse a missing, empty, repeated, or unknown argument, naming it. It reads only its arguments.
parse_packages_args() {
  local usage
  usage="$(packages_usage)"
  user_name=""
  scratch_root=""
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --user)
        need_value "$usage" "$@"
        [ -z "$user_name" ] || die "--user is given twice; $usage"
        user_name="$2"
        shift 2
        ;;
      --root)
        need_value "$usage" "$@"
        [ -z "$scratch_root" ] || die "--root is given twice; $usage"
        scratch_root="$2"
        shift 2
        ;;
      *) die "packages: unknown argument '$1'; $usage" ;;
    esac
  done
  [ -n "$user_name" ] || die "packages: --user is required and has no default; $usage"
  [ -n "$scratch_root" ] || die "packages: --root is required and has no default; $usage"
}

# check_disk_dir <dir>: refuse a disk directory that is not on C: or Z:, the drives Host Facts (Workstation scratch)
# allows for the distro's disk, judging the text alone: C:\..., C:/..., or Git Bash's /c/..., in either case.
check_disk_dir() {
  case "$1" in
    *[\\/]..[\\/]* | *[\\/]..) die "--disk-dir may not hold a '..' component, not '$1'" ;;
  esac
  case "$1" in
    [CcZz]:[\\/]?* | /[CcZz]/?*) ;;
    *) die "--disk-dir must be a directory on C: or Z: (Host Facts, Workstation scratch), such as C:\\WSL\\LASSI," \
      "not '$1'" ;;
  esac
}

# check_no_distro: refuse when a WSL distro named DISTRO already exists, read from `wsl.exe -l -q`. wsl.exe writes
# that list in UTF-16LE with CRLF line ends unless WSL_UTF8=1 reaches it, so the NUL and CR bytes (and a byte-order
# mark) are dropped before one line is matched exactly.
check_no_distro() {
  local list
  if ! list="$(WSL_UTF8=1 wsl.exe -l -q | tr -d '\000\r\376\377')"; then
    die "cannot list the WSL distros with wsl.exe -l -q"
  fi
  case $'\n'"$list"$'\n' in
    *$'\n'"$DISTRO"$'\n'*)
      die "a WSL distro named $DISTRO already exists (wsl.exe -l -q); the import stage makes a new one and never" \
        "changes an existing distro"
      ;;
  esac
}

# normalize_path <absolute path>: print the path with '.', '..', and repeated slashes resolved as text.
normalize_path() {
  local rest="$1/" part out=""
  while [ -n "$rest" ]; do
    part="${rest%%/*}"
    rest="${rest#*/}"
    case "$part" in
      '' | .) ;;
      ..) out="${out%/*}" ;;
      *) out="$out/$part" ;;
    esac
  done
  printf '%s\n' "${out:-/}"
}

# check_scratch_root <root>: refuse a scratch root that is relative, "/", or under /mnt (where WSL mounts the
# Windows drives), /tmp, /var/tmp, or /opt, judged after '..' is resolved as text and again after realpath -m, and
# one that holds a character tools/local/host-profile.sh refuses. The only place the script names those directories.
check_scratch_root() {
  local normal resolved path
  case "$1" in
    /*) ;;
    *) die "--root must be an absolute path, not '$1'" ;;
  esac
  case "$1" in
    *[!A-Za-z0-9._/-]*) die "--root may hold only letters, digits, '.', '_', '-', and '/', not '$1'" ;;
  esac
  normal="$(normalize_path "$1")"
  resolved="$(realpath -m -- "$normal")" || die "--root '$1' cannot be resolved with realpath -m"
  for path in "$normal" "$resolved"; do
    case "$path/" in
      // | /mnt/* | /tmp/* | /var/tmp/* | /opt/*)
        die "--root is '$1' ($path): the scratch root lies on the distro's own ext4 disk, never '/', /mnt (the" \
          "Windows drive mounts), /tmp, /var/tmp, or /opt (Host Facts, Workstation scratch)"
        ;;
    esac
  done
}

# check_euid: refuse unless this runs as root, read from `id -u`; the packages stage never calls sudo.
check_euid() {
  local uid
  uid="$(id -u)" || die "cannot read the user id with id -u"
  if [ "$uid" != 0 ]; then
    die "the packages stage runs as root inside the distro, started with wsl.exe -d $DISTRO -u root, and never" \
      "through sudo; id -u printed $uid"
  fi
}

# check_user_name <name>: refuse a name useradd would refuse, and root.
check_user_name() {
  if [ "$1" = root ] || [[ ! "$1" =~ ^[a-z_][a-z0-9_-]{0,31}$ ]]; then
    die "--user must be a plain lower-case Linux user name other than root, not '$1'"
  fi
}

# sha256_of <file>: print the file's sha256.
sha256_of() {
  local line
  line="$(sha256sum < "$1")" || return 1
  printf '%s\n' "${line%% *}"
}

# check_file <path> <size> <sha256>: succeed only when the file exists with that size in bytes and that sha256;
# otherwise say why on stderr and return 1.
check_file() {
  local actual=""
  if [ ! -f "$1" ]; then
    warn "$1: no such file"
    return 1
  fi
  if ! actual="$(stat -c %s -- "$1")" || [ "$actual" != "$2" ]; then
    warn "$1 has ${actual:-an unreadable number of} bytes, not the pinned $2"
    return 1
  fi
  if ! actual="$(sha256_of "$1")" || [ "$actual" != "$3" ]; then
    warn "$1 has sha256 ${actual:-that cannot be read}, not the pinned $3"
    return 1
  fi
}

# fetch <url> <destination> <size> <sha256>: download url over https into destination, unless a copy that checks
# out is there already, and refuse unless its size and sha256 are the pinned ones. A download that fails the check
# stays as <destination>.part for the owner to look at, and is never used.
fetch() {
  if [ -f "$2" ] && check_file "$2" "$3" "$4" 2> /dev/null; then
    say "kept $2, fetched before: $3 bytes, sha256 $4"
    return 0
  fi
  say "fetching $1"
  curl -q --fail --location --silent --show-error --proto =https --proto-redir =https --retry 3 \
    --max-filesize "$3" --output "$2.part" "$1" || die "could not fetch $1"
  check_file "$2.part" "$3" "$4" || die "refusing $1: it is not the file pinned on 2026-10-17"
  mv -f -- "$2.part" "$2"
  say "checked $2: $3 bytes, sha256 $4"
}

# check_drive_space <Git Bash path of the disk directory>: refuse when its drive has less free space than the floor
# plus this batch.
check_drive_space() {
  local drive="/${1:1:1}" free_kib need_kib
  free_kib="$(df -P -k "$drive" | awk 'NR == 2 { print $4 }')" || die "cannot read the free space of $drive with df"
  case "$free_kib" in
    '' | *[!0-9]*) die "df printed no free space for $drive" ;;
  esac
  need_kib=$(((DRIVE_FLOOR_GIB + BATCH_GIB) * 1024 * 1024))
  if [ "$free_kib" -lt "$need_kib" ]; then
    die "$drive has $((free_kib / 1048576)) GiB free; the floor is $DRIVE_FLOOR_GIB GiB (Host Facts, Workstation" \
      "scratch) and this batch needs about $BATCH_GIB GiB more (PROJECTED)"
  fi
  say "$drive has $((free_kib / 1048576)) GiB free: above the $DRIVE_FLOOR_GIB GiB floor plus $BATCH_GIB GiB" \
    "for this batch (PROJECTED)"
}

# import_stage <arguments>: stage 1, from Git Bash on Windows (see the top of the file).
import_stage() {
  local disk download
  parse_import_args "$@"
  check_disk_dir "$disk_dir"
  case "$download_dir" in
    [A-Za-z]:[\\/]* | /[A-Za-z]/*) ;;
    *) die "--download-dir must be an absolute Windows directory, such as C:\\Users\\<you>\\Downloads, not" \
      "'$download_dir'" ;;
  esac
  command -v cygpath > /dev/null || die "run the import stage from Git Bash on Windows; cygpath is not on PATH"
  command -v wsl.exe > /dev/null || die "wsl.exe is not on PATH"
  disk="$(cygpath -u "$disk_dir")"
  download="$(cygpath -u "$download_dir")"
  [ -d "$download" ] || die "--download-dir '$download_dir' is not an existing directory"
  [ ! -e "$disk/ext4.vhdx" ] || die "'$disk_dir' already holds an ext4.vhdx; give a new or empty directory"
  check_drive_space "$disk"
  check_no_distro
  wsl_script="/mnt$(cd "$(dirname -- "$script_path")" && pwd -P)/install-batch1.sh"
  run_import "$disk" "$download" 2>&1 | tee -a "$download/install-batch1-import.log"
}

# run_import <disk directory> <download directory>: fetch and check the image, then import it as DISTRO.
run_import() {
  local image="$2/${ROOTFS_URL##*/}" win_disk win_image
  export WSL_UTF8=1
  say "import stage, started $(date -u +%Y-%m-%dT%H:%M:%SZ); $script_path sha256 $(sha256_of "$script_path")"
  say "--disk-dir $disk_dir --download-dir $download_dir"
  fetch "$ROOTFS_URL" "$image" "$ROOTFS_SIZE" "$ROOTFS_SHA256"
  mkdir -p "$1"
  win_disk="$(cygpath -w "$1")"
  win_image="$(cygpath -w "$image")"
  say "importing $win_image as the WSL 2 distro $DISTRO, its disk in $win_disk"
  wsl.exe --import "$DISTRO" "$win_disk" "$win_image" --version 2 || die "wsl.exe --import failed (above)"
  wsl.exe -l -v
  say "imported $DISTRO, $(date -u +%Y-%m-%dT%H:%M:%SZ). The log is $2/install-batch1-import.log."
  say "next, the packages stage before any interactive start of the distro; $(packages_usage)"
}

# check_root_filesystem <root>: refuse a root whose nearest existing directory is not on an ext4 file system, the
# distro's own disk (stat -f prints ext2/ext3 for ext4).
check_root_filesystem() {
  local path="$1" type
  while [ ! -e "$path" ]; do
    path="$(dirname -- "$path")"
  done
  type="$(stat -f -c %T -- "$path")" || die "cannot read the file system of $path with stat -f"
  if [ "$type" != ext2/ext3 ]; then
    die "--root '$1' would lie on a $type file system (at $path), not on the distro's own ext4 disk"
  fi
}

# check_packages_host: refuse a host or a root this stage was not written for, before it changes anything.
check_packages_host() {
  local tool pin owner
  [ "${WSL_DISTRO_NAME:-}" = "$DISTRO" ] \
    || die "the packages stage runs in the distro $DISTRO (wsl.exe -d $DISTRO), not '${WSL_DISTRO_NAME:-}'"
  grep -qx 'VERSION_ID="22.04"' /etc/os-release || die "/etc/os-release does not name Ubuntu 22.04"
  for tool in curl gpg tar sha256sum stat awk realpath tee mktemp runuser; do
    command -v "$tool" > /dev/null || die "$tool, which the image ships, is not on PATH"
  done
  for pin in gcc tt-metal tt-mlir; do
    [ -f "$repo/toolchains/$pin.pin" ] || die "no $repo/toolchains/$pin.pin, which the version check reads"
  done
  if [ -e "$root" ]; then
    [ -d "$root" ] || die "--root '$root' exists and is not a directory"
    owner="$(stat -c %U -- "$root")"
    [ "$owner" = "$user_name" ] \
      || die "--root '$root' exists and belongs to $owner, not $user_name; give a new directory for the stage to make"
  fi
  check_root_filesystem "$root"
}

# packages_stage <arguments>: stage 2, inside the distro as root (see the top of the file).
packages_stage() {
  parse_packages_args "$@"
  check_euid
  check_scratch_root "$scratch_root"
  check_user_name "$user_name"
  export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
  root="$(realpath -m -- "$scratch_root")"
  repo="$(cd "$(dirname -- "$script_path")/../.." && pwd -P)"
  check_packages_host
  # The user first: for a root inside the user's home (--root /home/<user>/scratch), a directory made under the root
  # before useradd -m would leave that home owned by root, and useradd would copy no skeleton files into it.
  ensure_user
  mkdir -p "$root/install/batch1/debs" "$root/install/batch1/downloads" "$root/install/batch1/tmp"
  run_packages 2>&1 | tee -a "$root/install/batch1/install.log"
}

# run_packages: stage 2's changes after the user's creation, in order. give_install_tree follows the last step that
# writes under the root and comes before the version check, so the install record is the user's even when a tool
# differs from its pin.
run_packages() {
  export DEBIAN_FRONTEND=noninteractive
  export TMPDIR="$root/install/batch1/tmp"
  record_host start
  own_root
  write_wsl_conf
  add_llvm_repo
  install_apt_packages
  install_cmake
  install_uv
  give_install_tree
  check_versions
  record_host end
  say "done. Next, from PowerShell or cmd: wsl.exe --terminate $DISTRO, so that the distro starts again with" \
    "systemd and with $user_name as its default user."
}

# record_host <start or end>: log the date, the script and its arguments, the host, and the space used.
record_host() {
  say "packages stage, $1, $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if [ "$1" = start ]; then
    say "$script_path sha256 $(sha256_of "$script_path"); --user $user_name --root $root"
    say "kernel $(uname -srm)"
    cat /etc/os-release
  fi
  df -h / "$root"
  du -sh "$root"
}

# ensure_user: before any directory under the scratch root exists, create the user if missing (no password, no
# extra group), refuse uid 0 and the sudo and admin groups, and set user_group. The log does not exist yet, so what
# it found or did is kept in user_notes for own_root to log; a refusal goes to stderr.
ensure_user() {
  local groups
  user_notes=()
  if id -u "$user_name" > /dev/null 2>&1; then
    user_notes+=("user $user_name exists and is used as it is")
  else
    useradd -m -s /bin/bash "$user_name" || die "useradd could not create $user_name (above)"
    user_notes+=("created user $user_name: no password, no group beyond its own")
  fi
  [ "$(id -u "$user_name")" != 0 ] || die "user $user_name has uid 0"
  groups=" $(id -nG "$user_name") "
  case "$groups" in
    *" sudo "* | *" admin "*) die "user $user_name is in the sudo or admin group; this batch's user is not" ;;
  esac
  user_group="$(id -gn "$user_name")"
  user_notes+=("$(id "$user_name")")
}

# own_root: log what ensure_user found or did, now that the log exists, and give the scratch root and its bin to
# the user.
own_root() {
  local note
  for note in "${user_notes[@]}"; do
    say "$note"
  done
  chown "$user_name:$user_group" "$root"
  install -d -o "$user_name" -g "$user_group" -m 0755 "$root/bin"
  say "$root and $root/bin belong to $user_name"
}

# write_wsl_conf: write /etc/wsl.conf with systemd and the default user, keeping a copy of any earlier file.
write_wsl_conf() {
  local copy
  if [ -e /etc/wsl.conf ]; then
    copy="$root/install/batch1/wsl.conf.before-$(date -u +%Y%m%dT%H%M%SZ)"
    cp -p /etc/wsl.conf "$copy"
    say "kept the earlier /etc/wsl.conf as $copy:"
    cat "$copy"
  fi
  printf '%s\n' "# Written by tools/local/install-batch1.sh (task P5.15)." "[boot]" "systemd=true" "" "[user]" \
    "default=$user_name" > /etc/wsl.conf
  say "wrote /etc/wsl.conf:"
  cat /etc/wsl.conf
}

# add_llvm_repo: install apt.llvm.org's checked key as apt's keyring for it, and add the two jammy suites signed by
# it alone.
add_llvm_repo() {
  local keyring=/etc/apt/keyrings/lassi-llvm-snapshot.gpg list=/etc/apt/sources.list.d/lassi-llvm-jammy.list suite
  install_llvm_key "$keyring"
  for suite in $LLVM_APT_SUITES; do
    printf 'deb [signed-by=%s] %s %s main\n' "$keyring" "$LLVM_APT_REPO" "$suite"
  done > "$list"
  say "wrote $list:"
  cat "$list"
}

# install_llvm_key <keyring>: fetch apt.llvm.org's key and check its size, sha256, and fingerprint in a GnuPG home
# made for it under the root (root's own, mode 0700), dearmor it into the keyring, then remove that GnuPG home, which
# the user could not read.
install_llvm_key() {
  local key="$root/install/batch1/llvm-snapshot.gpg.key" gnupg fingerprint
  fetch "$LLVM_KEY_URL" "$key" "$LLVM_KEY_SIZE" "$LLVM_KEY_SHA256"
  gnupg="$(mktemp -d "$root/install/batch1/gnupg.XXXXXX")" || die "cannot make a GnuPG home in $root/install/batch1"
  fingerprint="$(GNUPGHOME="$gnupg" gpg --batch --show-keys --with-colons "$key" \
    | awk -F: '$1 == "fpr" && !seen { print $10; seen = 1 }')"
  [ "$fingerprint" = "$LLVM_KEY_FINGERPRINT" ] \
    || die "$key's primary key is '$fingerprint', not the pinned $LLVM_KEY_FINGERPRINT"
  say "checked the fingerprint of $key: $fingerprint"
  install -d -m 0755 /etc/apt/keyrings
  GNUPGHOME="$gnupg" gpg --batch --yes --dearmor --output "$1" "$key" || die "gpg could not dearmor $key"
  chmod 0644 "$1"
  say "dearmored $key into $1"
  remove_gnupg_home "$gnupg"
}

# remove_gnupg_home <directory>: remove the GnuPG home install_llvm_key made, and nothing else: refuse unless the
# root is absolute and the path is a directory, not a link, named gnupg. and six letters or digits (as mktemp -d
# names it) directly in <root>/install/batch1.
remove_gnupg_home() {
  case "$root" in
    /?*) ;;
    *) die "refusing to remove '$1': the scratch root '$root' is not an absolute path" ;;
  esac
  case "$1" in
    "$root/install/batch1/gnupg."[A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9]) ;;
    *) die "refusing to remove '$1': it is not a GnuPG home this stage makes in $root/install/batch1" ;;
  esac
  if [ ! -d "$1" ] || [ -L "$1" ]; then
    die "refusing to remove '$1': it is not a directory, or it is a link"
  fi
  rm -rf -- "$1"
  say "removed $1, the GnuPG home that checked the key"
}

# install_apt_packages: download every named package at its pinned version, with its dependencies, into the kept
# debs directory; check each named .deb; install from those files alone; and log dpkg-query -W.
install_apt_packages() {
  local debs="$root/install/batch1/debs" specs=() entry spec
  for entry in "${PACKAGES[@]}"; do
    specs+=("${entry%% *}")
  done
  apt-get update || die "apt-get update failed (above)"
  for spec in "${specs[@]}"; do
    apt-cache show "$spec" > /dev/null \
      || die "the indexes no longer serve $spec; re-read them and re-pin (plans/local-plan.md section 3.5)"
  done
  apt-get install --download-only --reinstall --yes --no-install-recommends --no-remove \
    -o Dir::Cache::archives="$debs" "${specs[@]}" \
    || die "apt-get could not download the packages (above); none is installed"
  check_named_debs "$debs"
  apt-get install --no-download --yes --no-install-recommends --no-remove -o Dir::Cache::archives="$debs" \
    -o APT::Keep-Downloaded-Packages=true "${specs[@]}" || die "apt-get could not install the packages (above)"
  (cd "$debs" && sha256sum -- *.deb) > "$root/install/batch1/debs.sha256"
  say "kept the .debs in $debs, their sha256 in $root/install/batch1/debs.sha256"
  say "dpkg-query -W, every installed package:"
  dpkg-query -W
  check_installed_versions
}

# check_named_debs <debs directory>: refuse unless each named .deb has the size and sha256 its index served.
check_named_debs() {
  local entry spec rest name version file failed=0
  for entry in "${PACKAGES[@]}"; do
    spec="${entry%% *}"
    rest="${entry#* }"
    name="${spec%%=*}"
    version="${spec#*=}"
    file="$1/${name}_${version//:/%3a}_amd64.deb"
    if check_file "$file" "${rest%% *}" "${rest#* }"; then
      say "checked $file: ${rest%% *} bytes, sha256 ${rest#* }"
    else
      failed=1
    fi
  done
  [ "$failed" = 0 ] || die "a named .deb is not the one its index served on 2026-10-17 (above); none is installed"
}

# check_installed_versions: refuse unless dpkg holds every named package at its pinned version.
check_installed_versions() {
  local entry spec actual failed=0
  for entry in "${PACKAGES[@]}"; do
    spec="${entry%% *}"
    actual="$(dpkg-query -W -f='${Version}' "${spec%%=*}")" || actual=""
    if [ "$actual" = "${spec#*=}" ]; then
      say "installed $spec"
    else
      warn "${spec%%=*} is installed at '$actual', not the pinned ${spec#*=}"
      failed=1
    fi
  done
  [ "$failed" = 0 ] || die "a named package is not at its pinned version (above)"
}

# install_cmake: fetch and check CMake's release tarball, list its files, and unpack it into /usr/local.
install_cmake() {
  local tarball="$root/install/batch1/downloads/${CMAKE_URL##*/}"
  fetch "$CMAKE_URL" "$tarball" "$CMAKE_SIZE" "$CMAKE_SHA256"
  tar -tzf "$tarball" > "$root/install/batch1/cmake-files.txt"
  tar -xzf "$tarball" -C /usr/local --strip-components=1 --no-same-owner --keep-directory-symlink \
    || die "could not unpack $tarball into /usr/local (above)"
  say "unpacked CMake $CMAKE_VERSION into /usr/local; its files are listed in $root/install/batch1/cmake-files.txt"
}

# install_uv: fetch and check uv's release tarball and put uv and uvx in <root>/bin, owned by the user.
install_uv() {
  local tarball="$root/install/batch1/downloads/${UV_URL##*/}" unpacked="$root/install/batch1/uv-$UV_VERSION"
  fetch "$UV_URL" "$tarball" "$UV_SIZE" "$UV_SHA256"
  mkdir -p "$unpacked"
  tar -xzf "$tarball" -C "$unpacked" --strip-components=1 --no-same-owner
  install -o "$user_name" -g "$user_group" -m 0755 "$unpacked/uv" "$unpacked/uvx" "$root/bin/"
  say "put uv and uvx $UV_VERSION in $root/bin, owned by $user_name"
}

# give_install_tree: give <root>/install, the kept .debs and this log included, to the user, links themselves and
# never what they point at (chown -R -P), so no directory under the root stays unreadable to the user (apt leaves
# debs/partial to _apt, mode 0700); then run du -sk of the scratch root as the user, the read tools/local/longjob.py
# makes before every job, and stop if it fails.
give_install_tree() {
  chown -R -P "$user_name:$user_group" "$root/install" || die "could not give $root/install to $user_name (above)"
  say "gave $root/install, its kept .debs and this log included, to $user_name"
  runuser -u "$user_name" -- du -sk "$root" || die "du -sk $root, run as $user_name, failed (above)"
}

# pin_value <pin file> <name>: print one value of a pin file, sourced in a clean bash as toolchains/*.sh source it.
pin_value() {
  env -i PATH=/usr/bin:/bin bash --noprofile --norc -c 'set -eu; source "$1"; printf "%s\n" "${!2}"' pin-value \
    "$1" "$2"
}

# expect_line <label> <expected> <program>: print the program's first --version line beside the expected string,
# and return 1 unless the line holds it, as toolchains/tt-mlir.sh's expect_version checks.
expect_line() {
  local out=""
  if ! out="$("$3" --version 2>&1)"; then
    warn "$3 --version failed: ${out%%$'\n'*}"
    return 1
  fi
  out="${out%%$'\n'*}"
  if [[ "$out" == *"$2"* ]]; then
    say "match   $3: '$out' | $1: '$2'"
  else
    warn "DIFFERS $3: '$out' | $1: '$2'"
    return 1
  fi
}

# check_pin <pin name> <tool variable>:<expected variable> ...: compare each tool the pin file names with the
# string the pin expects for it; print every pair, and return 1 if one differs.
check_pin() {
  local pin="$repo/toolchains/$1.pin" pair tool expected path failed=0
  for pair in "${@:2}"; do
    if ! tool="$(pin_value "$pin" "${pair%%:*}")" || ! expected="$(pin_value "$pin" "${pair#*:}")"; then
      warn "cannot read ${pair%%:*} and ${pair#*:} from $pin"
      failed=1
      continue
    fi
    if ! path="$(command -v -- "$tool")"; then
      warn "$pin names $tool (${pair%%:*}), which is not on PATH"
      failed=1
      continue
    fi
    expect_line "$1.pin ${pair#*:}" "$expected" "$path" || failed=1
  done
  return "$failed"
}

# check_versions: print each installed tool's version line beside the string its pin file expects, read from the
# pin files now, and fail if one differs.
check_versions() {
  local failed=0
  say "each tool's version line | the string its pin file expects:"
  check_pin gcc EXECUTABLE:EXPECT_VERSION || failed=1
  check_pin tt-metal EXECUTABLE:EXPECT_VERSION CLANG_C:CLANG_EXPECT CLANG_CXX:CLANG_EXPECT LINKER:LINKER_EXPECT \
    CMAKE:CMAKE_EXPECT NINJA:NINJA_EXPECT || failed=1
  check_pin tt-mlir CLANG_C:CLANG_EXPECT CLANG_CXX:CLANG_EXPECT LINKER:LINKER_EXPECT CMAKE:CMAKE_EXPECT \
    NINJA:NINJA_EXPECT PYTHON:PYTHON_EXPECT || failed=1
  expect_line "this batch's UV_VERSION" "uv $UV_VERSION" "$root/bin/uv" || failed=1
  [ "$failed" = 0 ] || die "a tool differs from its pin (above); a pin change is a Decision Log entry"
  say "every tool matches its pin"
}

# main <stage> <arguments>: run one stage.
main() {
  case "${1:-}" in
    import) shift; import_stage "$@" ;;
    packages) shift; packages_stage "$@" ;;
    '') die "name a stage, import or packages. $(import_usage). $(packages_usage)" ;;
    *) die "unknown stage '$1'; the stages are import and packages" ;;
  esac
}

script_path="${BASH_SOURCE[0]}"
main "$@"
