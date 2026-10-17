# shellcheck shell=bash
# The workstation's host profile (task P5.15; Host Facts, Workstation and Workstation scratch; AGENTS.md, Local
# Execution). Source it in bash inside the LASSI distro; never run it:
#
#   LASSI_LOCAL_ROOT=<scratch root> LASSI_LOCAL_DRIVE=</mnt/c or /mnt/z> LASSI_LOCAL_MEMORY_MAX=<size such as 20G> \
#     source tools/local/host-profile.sh
#
# It sets what alpha01's gate sets for every run (tools/server/gate.py, build_env), under the scratch root, so the
# toolchain scripts and the lassi package find the same environment on both hosts:
#   LASSI_SCRATCH (the root), LASSI_RUNS_ROOT, LASSI_TOOLCHAINS, LASSI_JOBS (8), TMPDIR, TMP, TEMP, the caches,
#   root/bin at the front of PATH, and PYTHONUNBUFFERED=1, with the gate's sub-paths
#   (tests/tools/test_local_profile.py reads them from gate.py, so the two cannot drift);
#   HOME at the root, since tt-mlir.sh checks HOME and git, CMake, and pip write under it;
#   LASSI_HOST=workstation, and the workstation's policy values: LASSI_ALLOWED_ROOT (the root, which the toolchain
#   scripts check their paths against), LASSI_STOP_GIB=145 and LASSI_CAP_GIB=150 (alpha01's numbers, so the
#   scripts keep one rule), LASSI_DRIVE_FLOOR_GIB=100 (free space kept on the drive that holds the distro's disk),
#   LASSI_HOST_DRIVE (that drive's mount), and LASSI_JOB_MEMORY_MAX (the memory cap of tools/local/longjob.py).
# It creates root/tmp, root/.cache, and root/bin, as build_env does, and defines lassi_nice, which runs its
# arguments under "ionice -c2 -n7 nice -n 10" (ionice only when present), as the gate's wrap does with nice 10.
#
# The three inputs have no default. The root must be an existing directory on the distro's own ext4 disk: an
# absolute path that is not "/" and lies under none of /mnt (where WSL mounts the Windows drives), /tmp, /var/tmp,
# or /opt, checked as given and after realpath -m. The profile checks the path only; tools/local/install-batch1.sh
# checks that the root's filesystem is ext4. A missing or bad input prints a message naming it to stderr and
# returns 2 with nothing set; the profile never exits the caller's shell, also under set -u.
# The profile sets no core limit: tools/local/longjob.py and the toolchain scripts hold the 1-byte limit for what
# they run, and a run outside them sets it itself (AGENTS.md, Local Execution).

if [ -z "${BASH_VERSION:-}" ]; then
  echo "host-profile: source this file from bash" >&2
  return 2 2>/dev/null || exit 2
fi
if [ "${BASH_SOURCE[0]:-}" = "${0:-}" ]; then
  echo "host-profile: source this file (source tools/local/host-profile.sh); running it sets nothing" >&2
  exit 2
fi

__lassi_profile_fail() {
  printf 'host-profile: %s\n' "$*" >&2
  return 2
}

# Print the checked scratch root (realpath -m of LASSI_LOCAL_ROOT) on stdout, or refuse.
__lassi_profile_root() {
  local given="${LASSI_LOCAL_ROOT:-}" resolved path
  if [ -z "$given" ]; then
    __lassi_profile_fail "LASSI_LOCAL_ROOT is not set: give the scratch root, an absolute path on the LASSI" \
      "distro's ext4 disk (Host Facts, Workstation scratch); it has no default"
    return 2
  fi
  case "$given" in
    /*) ;;
    *) __lassi_profile_fail "LASSI_LOCAL_ROOT must be an absolute path, not '$given'"; return 2 ;;
  esac
  case "$given" in
    *[!A-Za-z0-9._/-]*)
      __lassi_profile_fail "LASSI_LOCAL_ROOT may hold only letters, digits, '.', '_', '-', and '/', not '$given'"
      return 2 ;;
  esac
  if ! resolved="$(realpath -m -- "$given")" || [ -z "$resolved" ]; then
    __lassi_profile_fail "LASSI_LOCAL_ROOT cannot be resolved with realpath -m: '$given'"
    return 2
  fi
  for path in "$given" "$resolved"; do
    case "$path/" in
      // | /mnt/* | /tmp/* | /var/tmp/* | /opt/*)
        __lassi_profile_fail "LASSI_LOCAL_ROOT is '$given' ($resolved): the scratch root lies on the LASSI distro's" \
          "ext4 disk, never '/', /mnt (the Windows drive mounts), /tmp, /var/tmp, or /opt (Host Facts, Workstation" \
          "scratch)"
        return 2 ;;
    esac
  done
  if [ ! -d "$resolved" ]; then
    __lassi_profile_fail "LASSI_LOCAL_ROOT '$given' is not an existing directory; create the scratch root first"
    return 2
  fi
  printf '%s\n' "$resolved"
}

# Refuse a drive that is not the mount of C: or Z:, the drives the distro's disk may lie on.
__lassi_profile_drive() {
  local drive="${LASSI_LOCAL_DRIVE:-}"
  if [ -z "$drive" ]; then
    __lassi_profile_fail "LASSI_LOCAL_DRIVE is not set: give the mount of the Windows drive that holds the LASSI" \
      "distro's disk, /mnt/c or /mnt/z (Host Facts, Workstation scratch); it has no default"
    return 2
  fi
  case "$drive" in
    /mnt/c | /mnt/z) ;;
    *)
      __lassi_profile_fail "LASSI_LOCAL_DRIVE must be /mnt/c or /mnt/z, the mount of the Windows drive that holds" \
        "the LASSI distro's disk, not '$drive'"
      return 2 ;;
  esac
}

# Refuse a memory cap that is not a bounded systemd size: a positive whole number with K, M, G, or T.
__lassi_profile_memory() {
  local memory="${LASSI_LOCAL_MEMORY_MAX:-}"
  if [ -z "$memory" ]; then
    __lassi_profile_fail "LASSI_LOCAL_MEMORY_MAX is not set: give the long-job runner's memory cap, a systemd" \
      "size such as 20G (Host Facts, Workstation approvals); it has no default"
    return 2
  fi
  if ! [[ "$memory" =~ ^[1-9][0-9]*[KMGT]$ ]]; then
    __lassi_profile_fail "LASSI_LOCAL_MEMORY_MAX must be a positive whole number with K, M, G, or T, such as 20G," \
      "not '$memory'"
    return 2
  fi
}

# Export the environment under the checked root; the paths match gate.py's build_env.
__lassi_profile_export() {
  local root="$1"
  export LASSI_SCRATCH="$root"
  export LASSI_RUNS_ROOT="$root/lassi-runs"
  export LASSI_TOOLCHAINS="$root/toolchains"
  export LASSI_JOBS=8
  export TMPDIR="$root/tmp" TMP="$root/tmp" TEMP="$root/tmp"
  export XDG_CACHE_HOME="$root/.cache"
  export UV_CACHE_DIR="$root/.cache/uv"
  export PIP_CACHE_DIR="$root/.cache/pip"
  export HF_HOME="$root/hf"
  export TIKTOKEN_CACHE_DIR="$root/.cache/tiktoken"
  export CCACHE_DIR="$root/.cache/ccache"
  export CARGO_HOME="$root/.cargo"
  export RUSTUP_HOME="$root/.rustup"
  export DOTNET_CLI_HOME="$root/.dotnet-home"
  export NUGET_PACKAGES="$root/.nuget/packages"
  export npm_config_cache="$root/.cache/npm"
  case "${PATH:-}" in
    "$root/bin" | "$root/bin":*) ;;
    *) export PATH="$root/bin${PATH:+:$PATH}" ;;
  esac
  export PYTHONUNBUFFERED=1
  export HOME="$root"
  export LASSI_HOST=workstation
  export LASSI_ALLOWED_ROOT="$root"
  export LASSI_STOP_GIB=145
  export LASSI_CAP_GIB=150
  export LASSI_DRIVE_FLOOR_GIB=100
  export LASSI_HOST_DRIVE="$LASSI_LOCAL_DRIVE"
  export LASSI_JOB_MEMORY_MAX="$LASSI_LOCAL_MEMORY_MAX"
}

__lassi_profile_main() {
  local root
  root="$(__lassi_profile_root)" || return 2
  __lassi_profile_drive || return 2
  __lassi_profile_memory || return 2
  if ! mkdir -p -- "$root/tmp" "$root/.cache" "$root/bin"; then
    __lassi_profile_fail "cannot create tmp, .cache, and bin under the scratch root '$root'"
    return 2
  fi
  __lassi_profile_export "$root"
  # Run "$@" niced, as the gate's wrap does: best-effort I/O class 7 when ionice exists, and nice 10.
  lassi_nice() {
    if command -v ionice >/dev/null 2>&1; then
      ionice -c2 -n7 nice -n 10 "$@"
    else
      nice -n 10 "$@"
    fi
  }
}

if ! __lassi_profile_main; then
  unset -f __lassi_profile_fail __lassi_profile_root __lassi_profile_drive __lassi_profile_memory \
    __lassi_profile_export __lassi_profile_main
  return 2
fi
unset -f __lassi_profile_fail __lassi_profile_root __lassi_profile_drive __lassi_profile_memory \
  __lassi_profile_export __lassi_profile_main
