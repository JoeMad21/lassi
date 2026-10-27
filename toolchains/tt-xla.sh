#!/usr/bin/env bash
# Install the pinned tt-xla venv in user space (task P17.38), as recorded in toolchains/tt-xla.pin: the training stack
# the ttxla trainer runs its worker in (lassi.train.ttxla_trainer; plans/spikes/tt-training.md, Finding 3, step 1). On
# alpha01 it runs as its own big rx job, CPU only, before the owner's window or at the start of the window's sequence
# (tools/train_tt.py install, or plans/runs/train-tt/run.sh when the venv is missing), with
# LASSI_ALLOWED_ROOT=/mnt/nvme10/joseph_ufl and LASSI_STOP_GIB=345 on its command line; on the workstation the host
# profile (tools/local/host-profile.sh) sets them. Under $LASSI_TOOLCHAINS/$PREFIX_NAME it places:
#   downloads/   the CPython 3.11 archive (python-build-standalone, the build uv installs), checked by size and sha256;
#   python/      that interpreter, unpacked;
#   venv/        a venv of it made with uv, holding the tracked lock's packages, each checked by sha256;
#   wheels/      pjrt-plugin-tt, torch, and torch-xla, fetched by URL and checked by size and sha256 against the pin;
#   requirements.local.txt  the lock with those three lines pointed at wheels/.
# It refuses a wheel whose metadata names another tt-xla, tt-mlir, or tt-metal commit or another version, and links
# the pinned sfpi 7.25.0 of the tt-metal install at the pin into the wheel's runtime directory, since the wheel
# carries no sfpi and the pinned tt-metal would otherwise take the host's /opt/tenstorrent/sfpi, another version
# (the spike, Finding 1c). Nothing touches a device: no Python of the venv imports a framework, and no device
# variable is set; the one program of the venv it runs reads package metadata. Nothing is written outside the
# allowed root (Agent Rule 7): no sudo, no system package, the downloads, caches, and temporary files under the
# scratch root. The prefix carries lassi-install.unfinished until every check has passed, and only then is that file
# renamed to the install record, lassi-install.txt. Idempotent: an install whose record matches the pin and that
# checks out is kept; an unfinished prefix of the same pin is resumed; anything else at the prefix is refused and
# left as it is, since agents never delete on the build host.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=tt-xla.pin
source "$here/tt-xla.pin"
# No value below has a default (AGENTS.md, rules for [OPEN] choices). On alpha01 the gate sets LASSI_TOOLCHAINS and
# LASSI_SCRATCH, and the rx command line gives LASSI_ALLOWED_ROOT and LASSI_STOP_GIB; HOME is the gate's own, the
# scratch root on the build host.
set_by="by tools/local/host-profile.sh on the workstation, or on alpha01 by the gate or the rx command line"
: "${LASSI_TOOLCHAINS:?LASSI_TOOLCHAINS must be set $set_by}"
: "${LASSI_SCRATCH:?LASSI_SCRATCH must be set $set_by}"
: "${LASSI_ALLOWED_ROOT:?LASSI_ALLOWED_ROOT must be set $set_by}"
: "${LASSI_STOP_GIB:?LASSI_STOP_GIB must be set $set_by}"
: "${HOME:?HOME must be set; the gate passes its own HOME through, the scratch root on the build host}"
case "$LASSI_ALLOWED_ROOT" in
  / | [!/]*)
    echo "tt-xla: LASSI_ALLOWED_ROOT, set $set_by, must be an absolute path other than /, not '$LASSI_ALLOWED_ROOT'" >&2
    exit 2
    ;;
esac
for root in "$LASSI_TOOLCHAINS" "$LASSI_SCRATCH" "$HOME"; do
  case "$(realpath -m "$root")/" in
    "$LASSI_ALLOWED_ROOT"/*) ;;
    *)
      echo "tt-xla: refusing to work outside $LASSI_ALLOWED_ROOT, the LASSI_ALLOWED_ROOT set $set_by (Agent Rule" \
        "7): $root" >&2
      exit 2
      ;;
  esac
done
case "$LASSI_STOP_GIB" in
  '' | *[!0-9]* | 0*)
    echo "tt-xla: LASSI_STOP_GIB, the stop line in GiB, must be a positive integer with no leading 0, not" \
      "'$LASSI_STOP_GIB'" >&2
    exit 2
    ;;
esac
# A crashing tool must leave no core (Agent Rule 7): the kernel aborts a piped core dump only at a limit of exactly 1
# byte (plans/LESSONS.md, alpha01), so prlimit sets it in bytes, soft and hard, and Python's getrlimit reads it back as
# (1, 1); every child inherits it. It comes after the root checks and before anything is downloaded.
if ! prlimit --pid $$ --core=1:1 \
  || ! python3 -I -c 'import resource, sys; sys.exit(resource.getrlimit(resource.RLIMIT_CORE) != (1, 1))'; then
  echo "tt-xla: cannot hold RLIMIT_CORE at exactly 1 byte, soft and hard (prlimit, read back as (1, 1)); a crashing" \
    "tool could store a core on the root filesystem (Agent Rule 7)" >&2
  exit 2
fi
prefix="$LASSI_TOOLCHAINS/$PREFIX_NAME"
export TMPDIR="$LASSI_SCRATCH/tmp"
mkdir -p "$TMPDIR"
# The host environment adds nothing: no Python, pip, or uv setting, no tt-metal variable, and no uv configuration file
# (UV_NO_CONFIG); uv never downloads an interpreter (UV_PYTHON_DOWNLOADS=never) and keeps no cache beyond the run.
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE VIRTUAL_ENV CONDA_PREFIX LD_LIBRARY_PATH
for name in $(compgen -e); do
  case "$name" in
    UV_* | PIP_* | TT_METAL_* | TT_XLA_*) unset "$name" ;;
  esac
done
export UV_NO_CONFIG=1 UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy UV_CACHE_DIR="$LASSI_SCRATCH/.cache/uv-tt-xla"
record_name=lassi-install.txt
unfinished_name=lassi-install.unfinished
host_name=lassi-install-host.txt
venv="$prefix/$VENV_DIR"
python_bin="$prefix/$PYTHON_DIR/bin/python$PYTHON_MINOR"
site="$venv/lib/python$PYTHON_MINOR/site-packages"
metal="$LASSI_TOOLCHAINS/$METAL_PREFIX_NAME"

# pinned_record: print the record an install from the current pin carries: the wheel first (lassi.train.ttxla_trainer
# reads that line), every URL, size, and sha256, the lock's sha256, the commits, the sfpi link, and the layout. It
# runs no tool but sha256sum and writes nothing.
pinned_record() {
  local digest
  digest="$(sha256sum <"$here/$PYTHON_REQUIREMENTS")"
  echo "$NAME $VERSION $WHEEL_SHA256"
  echo "wheel $WHEEL_URL $WHEEL_SIZE"
  echo "torch $TORCH_URL $TORCH_SHA256 $TORCH_SIZE"
  echo "torch-xla $TORCH_XLA_URL $TORCH_XLA_SHA256 $TORCH_XLA_SIZE"
  echo "python $PYTHON_VERSION $PYTHON_URL $PYTHON_SHA256 $PYTHON_SIZE"
  echo "requirements $PYTHON_REQUIREMENTS ${digest%% *} from $INDEX_URL"
  echo "commits tt-xla $TT_XLA_COMMIT tt-mlir $TT_MLIR_COMMIT tt-metal $TT_METAL_COMMIT"
  echo "sfpi $SFPI_VERSION $SFPI_SHA256 linked from $METAL_PREFIX_NAME/runtime/sfpi"
  echo "layout python $PYTHON_DIR; venv $VENV_DIR; wheels $WHEELS_DIR; downloads $DOWNLOADS_DIR"
}

# check_prefix: pass when nothing is at the prefix, or when it holds an unfinished prefix of the same pin, which the
# install resumes; refuse anything else and leave it as it is.
check_prefix() {
  if [ ! -e "$prefix" ] && [ ! -L "$prefix" ]; then
    echo "tt-xla: nothing is at $prefix yet"
    return 0
  fi
  if [ ! -L "$prefix" ] && [ -f "$prefix/$unfinished_name" ] && [ ! -e "$prefix/$record_name" ] \
    && [ "$(cat "$prefix/$unfinished_name")" = "$(pinned_record)" ]; then
    echo "tt-xla: resuming the unfinished prefix at $prefix, which has the same pin"
    return 0
  fi
  echo "tt-xla: $prefix holds an install or unfinished prefix of another pin, or unknown content; it is refused and" \
    "left as it is, since agents never delete on the build host. Ask the owner (plans/OWNER-QUEUE.md)." >&2
  return 1
}

# check_host: refuse when a tool the install runs is missing; print each one's version for the log.
check_host() {
  local tool
  for tool in uv curl tar sha256sum realpath; do
    if ! command -v "$tool" >/dev/null; then
      echo "tt-xla: $tool, which the install runs, is not on PATH" >&2
      return 1
    fi
  done
  echo "tt-xla: $(uv --version); $(curl --version | head -1); $(tar --version | head -1)"
}

# check_metal: refuse unless the tt-metal install at the pin is there (its record's first line names tt-metal and
# TT_METAL_COMMIT) with its sfpi compiler reporting SFPI_VERSION, which the wheel's runtime directory will link to.
check_metal() {
  local first out
  first="$(head -1 "$metal/lassi-install.txt" 2>/dev/null || true)"
  if [ "$(echo "$first" | cut -d' ' -f1-2)" != "tt-metal $TT_METAL_COMMIT" ]; then
    echo "tt-xla: $metal is not the tt-metal install at $TT_METAL_COMMIT (its record begins '$first'); install it" \
      "with toolchains/tt-metal.sh first" >&2
    return 1
  fi
  if ! out="$("$metal/runtime/sfpi/compiler/bin/riscv-tt-elf-g++" --version 2>&1 | head -1)"; then
    echo "tt-xla: cannot run the sfpi compiler of $metal" >&2
    return 1
  fi
  case "$out" in
    *":${SFPI_VERSION})"* | *":${SFPI_VERSION}["*) echo "tt-xla: $metal has sfpi $SFPI_VERSION ($out)" ;;
    *) echo "tt-xla: $metal's sfpi compiler reports '$out', not $SFPI_VERSION" >&2; return 1 ;;
  esac
}

# check_space <planned KiB>: refuse when the planned bytes do not fit the scratch disk's free space, or would pass the
# stop line, LASSI_STOP_GIB GiB, added to what the scratch root holds now (du -s), as toolchains/tt-mlir.sh does.
check_space() {
  local planned="$1" stop=$((LASSI_STOP_GIB * 1024 * 1024)) used free value
  used="$( (du -sk "$LASSI_SCRATCH" 2>/dev/null || true) | cut -f1)"
  free="$(df -Pk "$LASSI_SCRATCH" | awk 'NR == 2 { print $4 }')"
  for value in "$used" "$free"; do
    case "$value" in
      '' | *[!0-9]*) echo "tt-xla: cannot measure the scratch disk (du '$used', df '$free')" >&2; return 1 ;;
    esac
  done
  if [ $((used + planned)) -gt "$stop" ] || [ "$planned" -gt "$free" ]; then
    echo "tt-xla: $LASSI_SCRATCH holds $used KiB and $free KiB are free; the planned $planned KiB would pass the" \
      "$LASSI_STOP_GIB GiB stop line ($stop KiB) or the free space" >&2
    return 1
  fi
  echo "tt-xla: space check passed: $used KiB used, $planned KiB planned, stop line $stop KiB"
}

# check_download <file> <size> <sha256>: refuse unless the file has exactly that size in bytes and that sha256.
check_download() {
  local file="$1" size="$2" sha="$3" found digest
  found="$(wc -c <"$file" 2>/dev/null | tr -d ' ' || true)"
  digest="$(sha256sum <"$file" 2>/dev/null || true)"
  if [ "$found" != "$size" ] || [ "${digest%% *}" != "$sha" ]; then
    echo "tt-xla: $file has $found bytes and sha256 ${digest%% *}, not the pinned $size and $sha" >&2
    return 1
  fi
}

# decoded <name>: print a URL's file name with %2B written + (the wheels' local versions), as a file name on disk.
decoded() {
  echo "${1//%2B/+}"
}

# fetch_checked <url> <file> <size> <sha256>: keep a file already there with the pinned size and sha256; otherwise
# download the URL to <file>.part, check it, and rename it. A download that fails the check is removed and refused.
fetch_checked() {
  local url="$1" file="$2" size="$3" sha="$4"
  if [ -f "$file" ] && check_download "$file" "$size" "$sha" 2>/dev/null; then
    echo "tt-xla: $file is already there with the pinned size and sha256"
    return 0
  fi
  if [ -e "$file" ]; then
    echo "tt-xla: $file is there with another size or sha256; it is refused and left as it is" >&2
    return 1
  fi
  curl -fsSL --retry 3 --proto '=https' -o "$file.part" "$url"
  if ! check_download "$file.part" "$size" "$sha"; then
    rm -f "$file.part"
    return 1
  fi
  mv -T "$file.part" "$file"
  echo "tt-xla: fetched $url ($size bytes, sha256 $sha)"
}

# unpack_python: unpack the interpreter archive into PYTHON_DIR unless an interpreter of PYTHON_VERSION is there.
unpack_python() {
  local archive="$prefix/$DOWNLOADS_DIR/$(decoded "${PYTHON_URL##*/}")"
  if [ ! -x "$python_bin" ]; then
    tar -xzf "$archive" -C "$prefix" --no-same-owner
  fi
  check_python "$python_bin"
}

# check_python <interpreter>: refuse unless it reports Python PYTHON_VERSION.
check_python() {
  local out
  out="$("$1" --version 2>&1 || true)"
  if [ "$out" != "Python $PYTHON_VERSION" ]; then
    echo "tt-xla: $1 reports '$out', not Python $PYTHON_VERSION" >&2
    return 1
  fi
  echo "tt-xla: $1: $out"
}

# make_venv: make the venv of the unpacked interpreter with uv unless it is there.
make_venv() {
  if [ ! -x "$venv/bin/python" ]; then
    uv venv --no-cache --python "$python_bin" "$venv"
  fi
  check_python "$venv/bin/python"
}

# local_requirements <tracked lock> <out> <wheel dir>: write the lock with its three direct-URL lines pointed at the
# fetched wheels in <wheel dir> (file:// URLs keeping the %2B escapes; the files on disk carry +), every hash kept.
local_requirements() {
  local tracked="$1" out="$2" wheels="$3" text url
  text="$(cat "$tracked")"
  for url in "$WHEEL_URL" "$TORCH_URL" "$TORCH_XLA_URL"; do
    if [ "$(grep -cF " @ $url" "$tracked")" != 1 ]; then
      echo "tt-xla: $tracked does not name $url exactly once; the lock and the pin disagree" >&2
      return 1
    fi
    text="${text//" @ $url"/" @ file://$wheels/${url##*/}"}"
  done
  printf '%s\n' "$text" >"$out"
}

# install_packages: install the local lock into the venv: every distribution checked against its sha256, none resolved
# again or built from source, the index only for the ones not fetched above.
install_packages() {
  uv pip install --no-cache --python "$venv/bin/python" --require-hashes --no-deps --only-binary :all: \
    --index-url "$INDEX_URL" -r "$prefix/requirements.local.txt"
}

# check_wheel_metadata <METADATA>: refuse unless the wheel's metadata has the pinned version and a Summary naming the
# pinned tt-xla, tt-mlir, and tt-metal commits, each as its own comma-separated part.
check_wheel_metadata() {
  local file="$1" version summary part want found
  local -a parts
  version="$(sed -n 's/^Version: //p' "$file" | head -1)"
  summary="$(sed -n 's/^Summary: //p' "$file" | head -1)"
  if [ "$version" != "$VERSION" ]; then
    echo "tt-xla: $file names version '$version', not the pinned $VERSION" >&2
    return 1
  fi
  for want in "commit=$TT_XLA_COMMIT" "tt-mlir-commit=$TT_MLIR_COMMIT" "tt-metal-commit=$TT_METAL_COMMIT"; do
    found=0
    IFS=',' read -r -a parts <<<"$summary"
    for part in "${parts[@]}"; do
      if [ "$(echo "$part" | tr -d ' ')" = "$want" ]; then found=1; fi
    done
    if [ "$found" != 1 ]; then
      echo "tt-xla: $file's Summary '$summary' does not name $want; a wheel of another tt-xla, tt-mlir, or" \
        "tt-metal commit is refused (Agent Rule 10)" >&2
      return 1
    fi
  done
  echo "tt-xla: the wheel's metadata names $VERSION and the pinned tt-xla, tt-mlir, and tt-metal commits"
}

# check_venv: refuse unless every distribution of the lock is installed at its locked version, read from the
# installed metadata by the venv's interpreter (importlib.metadata only; no framework is imported).
check_venv() {
  "$venv/bin/python" -I - "$here/$PYTHON_REQUIREMENTS" <<'PY'
import importlib.metadata as metadata
import re
import sys

text = re.sub(r"\\\n\s*", " ", open(sys.argv[1], encoding="ascii").read())
missing = []
for line in text.splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    name = line.split()[0].split("==")[0]
    if " @ " in line:
        wanted = line.split(" @ ")[1].split()[0].rsplit("/", 1)[1].replace("%2B", "+").split("-")[1]
    else:
        wanted = line.split()[0].split("==")[1]
    try:
        found = metadata.version(name)
    except metadata.PackageNotFoundError:
        found = None
    if found != wanted:
        missing.append(f"{name} {wanted} (found {found})")
if missing:
    sys.exit("tt-xla: the venv lacks these pins of the lock: " + "; ".join(missing))
print("tt-xla: the venv holds every pin of the lock")
PY
}

# check_sfpi_version_file <sfpi-version>: refuse unless the wheel's tt-metal names the pinned sfpi and its txz sha256.
check_sfpi_version_file() {
  local file="$1"
  if ! grep -qxF "sfpi_version='$SFPI_VERSION'" "$file" \
    || ! grep -qxF "sfpi_x86_64_debian_txz_hash='$SFPI_SHA256'" "$file"; then
    echo "tt-xla: $file does not name sfpi $SFPI_VERSION with sha256 $SFPI_SHA256" >&2
    return 1
  fi
  echo "tt-xla: the wheel's tt-metal names sfpi $SFPI_VERSION"
}

# link_sfpi: link the pinned install's runtime/sfpi into the wheel's runtime directory, where the pinned tt-metal
# looks first (tt_metal/jit_build/build.cpp:140-154 at the pin); keep a link already there to it, refuse anything else.
link_sfpi() {
  local runtime="$site/pjrt_plugin_tt/tt-metal/runtime" target="$metal/runtime/sfpi" out
  check_sfpi_version_file "$site/pjrt_plugin_tt/tt-metal/tt_metal/sfpi-version"
  if [ -L "$runtime/sfpi" ] && [ "$(readlink "$runtime/sfpi")" = "$target" ]; then
    echo "tt-xla: $runtime/sfpi already links to $target"
  elif [ -e "$runtime/sfpi" ] || [ -L "$runtime/sfpi" ]; then
    echo "tt-xla: $runtime/sfpi is there and is not a link to $target; it is refused and left as it is" >&2
    return 1
  else
    ln -s "$target" "$runtime/sfpi"
  fi
  out="$("$runtime/sfpi/compiler/bin/riscv-tt-elf-g++" --version 2>&1 | head -1)"
  case "$out" in
    *":${SFPI_VERSION})"* | *":${SFPI_VERSION}["*) echo "tt-xla: the wheel's runtime takes sfpi $SFPI_VERSION" ;;
    *) echo "tt-xla: the linked sfpi compiler reports '$out', not $SFPI_VERSION" >&2; return 1 ;;
  esac
}

# verify_install: the checks a kept install must pass: the interpreter, the venv, the wheel, every pin, and the link.
verify_install() {
  check_python "$python_bin" && check_python "$venv/bin/python" \
    && check_wheel_metadata "$(wheel_metadata)" && check_venv && link_sfpi
}

# wheel_metadata: print the path of the installed wheel's METADATA file.
wheel_metadata() {
  echo "$site/pjrt_plugin_tt-$VERSION.dist-info/METADATA"
}

# check_root_fs <marker> <dir>...: fail when this user changed anything under the directories since <marker>, without
# crossing into other file systems (Agent Rule 7); the method of toolchains/tt-mlir.sh, whose comment explains it.
check_root_fs() {
  local marker="$1" dir out line status=0 denied=0 leaked="" failed=""
  shift
  for dir in "$@"; do
    if [ ! -d "$dir" ]; then
      echo "tt-xla: $dir is not a directory, so the root-filesystem check cannot run; nothing is recorded" >&2
      return 1
    fi
  done
  out="$(LC_ALL=C find "$@" -xdev -user "$(id -un)" -newer "$marker" -printf 'changed %p\n' 2>&1)" || status=$?
  while IFS= read -r line; do
    case "$line" in
      '') ;;
      'changed '*) leaked+="${line#changed }"$'\n' ;;
      'find: '*': Permission denied') denied=$((denied + 1)) ;;
      *) failed+="$line"$'\n' ;;
    esac
  done <<<"$out"
  if [ -n "$failed" ] || { [ "$status" -ne 0 ] && [ "$denied" -eq 0 ]; }; then
    echo "tt-xla: the root-filesystem check over $* did not run cleanly (find rc=$status); nothing is recorded" >&2
    printf '%s' "$failed" >&2
    return 1
  fi
  if [ -n "$leaked" ]; then
    echo "tt-xla: files changed under $* during the install; nothing is recorded:" >&2
    printf '%s' "$leaked" >&2
    return 1
  fi
  echo "tt-xla: root-filesystem check passed (find rc=$status, $denied unreadable paths skipped)"
}

# Everything this run creates outside the prefix, and nothing else, is removed on exit: its start marker.
marker=""
cleanup() {
  if [ -n "$marker" ]; then rm -f "$marker"; fi
}
trap cleanup EXIT

if [ -f "$prefix/$record_name" ] && [ "$(cat "$prefix/$record_name")" = "$(pinned_record)" ]; then
  if ! verify_install; then
    echo "tt-xla: $prefix carries the pinned record but does not check out; it is left as it is: inspect it, and ask" \
      "the owner before anything there is moved or removed" >&2
    exit 1
  fi
  echo "tt-xla: $prefix already holds the pinned tt-xla $VERSION"
  exit 0
fi

marker="$(mktemp "$TMPDIR/tt-xla-start.XXXXXX")"
# Every refusal comes before the first download: what is at the prefix, the tools, the tt-metal install, the space.
check_prefix
check_host
check_metal
check_space "$INSTALL_KIB"
if [ ! -e "$prefix" ]; then
  mkdir -p "$LASSI_TOOLCHAINS"
  mkdir "$prefix"
  pinned_record >"$prefix/$unfinished_name"
fi
mkdir -p "$prefix/$DOWNLOADS_DIR" "$prefix/$WHEELS_DIR"
fetch_checked "$PYTHON_URL" "$prefix/$DOWNLOADS_DIR/$(decoded "${PYTHON_URL##*/}")" "$PYTHON_SIZE" "$PYTHON_SHA256"
unpack_python
make_venv
fetch_checked "$WHEEL_URL" "$prefix/$WHEELS_DIR/$(decoded "${WHEEL_URL##*/}")" "$WHEEL_SIZE" "$WHEEL_SHA256"
fetch_checked "$TORCH_URL" "$prefix/$WHEELS_DIR/$(decoded "${TORCH_URL##*/}")" "$TORCH_SIZE" "$TORCH_SHA256"
fetch_checked "$TORCH_XLA_URL" "$prefix/$WHEELS_DIR/$(decoded "${TORCH_XLA_URL##*/}")" "$TORCH_XLA_SIZE" \
  "$TORCH_XLA_SHA256"
local_requirements "$here/$PYTHON_REQUIREMENTS" "$prefix/requirements.local.txt" "$prefix/$WHEELS_DIR"
install_packages
check_wheel_metadata "$(wheel_metadata)"
check_venv
link_sfpi
# Nothing may land on the root filesystem (Agent Rule 7).
check_root_fs "$marker" /tmp /var/tmp /opt
{
  echo "uv $(uv --version)"
  echo "installed $(date -u +%Y-%m-%dT%H:%M:%SZ) on $(hostname)"
} >"$prefix/$host_name"
mv -T "$prefix/$unfinished_name" "$prefix/$record_name"
echo "tt-xla: installed $PREFIX_NAME in $prefix"
du -sk "$prefix"
du -sk "$LASSI_SCRATCH"
