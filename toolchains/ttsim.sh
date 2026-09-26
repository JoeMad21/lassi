#!/usr/bin/env bash
# Install the pinned ttsim in user space (task P4.2), as recorded in toolchains/ttsim.pin: the release's
# Wormhole library, checked against its pinned size and sha256, with the Wormhole soc descriptor from the
# pinned tt-metal tree beside it, under the name tt-metal's own ttsim CI job gives it. Run on the build host
# through `uv run tools/rx.py job start --big`, after toolchains/tt-metal.sh. Nothing here loads the library or
# runs a program, so nothing opens a device.
# Idempotent: an install whose record and files match the pin is kept, and a verified download is reused. A new
# install is assembled in a staging prefix and checked; only then is a previous install moved aside (never
# deleted) and the staging prefix renamed into place.
set -euo pipefail
# A tool that crashes leaves no core file, which the host's core handler would put on the root filesystem
# (Agent Rule 7).
ulimit -c 0
here="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=ttsim.pin
source "$here/ttsim.pin"
: "${LASSI_TOOLCHAINS:?LASSI_TOOLCHAINS must be set; the gate sets it}"
: "${LASSI_SCRATCH:?LASSI_SCRATCH must be set; the gate sets it}"
for root in "$LASSI_TOOLCHAINS" "$LASSI_SCRATCH"; do
  case "$(realpath -m "$root")/" in
    /mnt/nvme10/joseph_ufl/*) ;;
    *) echo "ttsim: refusing to work outside /mnt/nvme10/joseph_ufl (Agent Rule 7): $root" >&2; exit 2 ;;
  esac
done
# curl and bash for here-documents put temporary files under TMPDIR; keep them on the scratch disk
# (Agent Rule 7).
export TMPDIR="$LASSI_SCRATCH/tmp"
mkdir -p "$TMPDIR"
prefix="$LASSI_TOOLCHAINS/$PREFIX_NAME"
staging="$prefix.staging"
record_name=lassi-install.txt
# The pinned tt-metal tree the descriptor is copied from: tt-metal.pin is read in a subshell so that its keys
# do not replace this pin's.
# shellcheck source=tt-metal.pin
read -r tt_metal_prefix tt_metal_commit <<<"$(source "$here/tt-metal.pin" && echo "$PREFIX_NAME $COMMIT")"
if [ "$tt_metal_prefix" != "$TT_METAL_FROM" ]; then
  echo "ttsim: ttsim.pin copies from $TT_METAL_FROM, but tt-metal.pin installs $tt_metal_prefix" >&2
  exit 1
fi
tt_metal="$LASSI_TOOLCHAINS/$TT_METAL_FROM"

# pinned_record: print the record an install from the current pin carries.
pinned_record() {
  echo "ttsim $VERSION $URL $SIZE $SHA256"
  echo "soc_descriptor $SOC_DESCRIPTOR from $TT_METAL_FROM/$SOC_DESCRIPTOR_FROM $SOC_DESCRIPTOR_SHA256"
}

# has_sha256 <file> <sha256>: succeed when <file> is a file with that sha256.
has_sha256() {
  [ -f "$1" ] && echo "$2  $1" | sha256sum -c --status
}

# installed_matches <dir>: succeed when <dir> holds this pin's record, the library with its pinned size and
# sha256, and the soc descriptor with its pinned sha256.
installed_matches() {
  local dir="$1"
  [ -f "$dir/$record_name" ] && [ "$(cat "$dir/$record_name")" = "$(pinned_record)" ] \
    && [ "$(stat -c %s "$dir/$LIBRARY" 2>/dev/null || true)" = "$SIZE" ] \
    && has_sha256 "$dir/$LIBRARY" "$SHA256" \
    && has_sha256 "$dir/$SOC_DESCRIPTOR" "$SOC_DESCRIPTOR_SHA256"
}

if installed_matches "$prefix"; then
  echo "ttsim: $prefix already holds the pinned ttsim $VERSION"
  sha256sum "$prefix/$LIBRARY" "$prefix/$SOC_DESCRIPTOR"
  exit 0
fi

# check_space <planned KiB>: refuse when the planned bytes do not fit in the free space of the scratch disk,
# or when, added to what the scratch root holds now (du -s), they would pass the P4 plan's stop line of
# 115 GiB, which keeps the scratch root under the owner's cap of 120 GiB (plans/p4-ttsim.md).
check_space() {
  local planned="$1" stop=$((115 * 1024 * 1024)) used free value
  used="$( (du -sk "$LASSI_SCRATCH" 2>/dev/null || true) | cut -f1)"
  free="$(df -Pk "$LASSI_SCRATCH" | awk 'NR == 2 { print $4 }')"
  for value in "$used" "$free"; do
    case "$value" in
      '' | *[!0-9]*)
        echo "ttsim: cannot measure the scratch disk (du -s gave '$used' KiB, df gave '$free' KiB free)" >&2
        return 1
        ;;
    esac
  done
  if [ $((used + planned)) -gt "$stop" ]; then
    echo "ttsim: $LASSI_SCRATCH holds $used KiB; the planned $planned KiB would pass the 115 GiB stop line" \
      "($stop KiB)" >&2
    echo "ttsim: queue backup or deletion candidates in the owner queue before installing" >&2
    return 1
  fi
  if [ "$planned" -gt "$free" ]; then
    echo "ttsim: the planned $planned KiB do not fit in the $free KiB free on the scratch disk" >&2
    return 1
  fi
  echo "ttsim: space check passed: $used KiB used under $LASSI_SCRATCH, $planned KiB planned, stop line $stop KiB"
}

# fetch <url> <sha256> <file>: keep <file> when it already has the sha256; otherwise download it to a new
# temporary name next to it, check the sha256, and rename it into place. A file that exists with another
# sha256 is refused and left as it is, since this run did not create it. curl -q, first, reads no curlrc.
fetch() {
  local url="$1" sum="$2" file="$3" part
  if [ -e "$file" ]; then
    if echo "$sum  $file" | sha256sum -c --status; then
      echo "ttsim: reusing $file (sha256 $sum)"
      return 0
    fi
    echo "ttsim: $file exists but its sha256 is not $sum; inspect it, then rerun" >&2
    return 1
  fi
  part="$(mktemp "$file.part.XXXXXX")"
  if ! curl -q -fsSL --retry 3 -o "$part" "$url"; then
    rm -f "$part"
    echo "ttsim: cannot download $url" >&2
    return 1
  fi
  if ! echo "$sum  $part" | sha256sum -c --status; then
    rm -f "$part"
    echo "ttsim: $url does not have the sha256 $sum" >&2
    return 1
  fi
  mv "$part" "$file"
  echo "ttsim: downloaded $file (sha256 $sum)"
}

# source_descriptor: print the path of the soc descriptor in the pinned tt-metal tree after checking that the
# tree is at its pinned commit and the file has its pinned sha256; refuse otherwise.
source_descriptor() {
  local head file="$tt_metal/$SOC_DESCRIPTOR_FROM"
  if [ ! -e "$tt_metal/.git" ]; then
    echo "ttsim: $tt_metal is not a checkout; run toolchains/tt-metal.sh first" >&2
    return 1
  fi
  head="$(git --git-dir="$tt_metal/.git" rev-parse HEAD || true)"
  if [ "$head" != "$tt_metal_commit" ]; then
    echo "ttsim: $tt_metal is at '$head', not the pinned $tt_metal_commit" >&2
    return 1
  fi
  if ! has_sha256 "$file" "$SOC_DESCRIPTOR_SHA256"; then
    echo "ttsim: $file does not have the pinned sha256 $SOC_DESCRIPTOR_SHA256" >&2
    return 1
  fi
  echo "$file"
}

# check_root_fs <marker> <dir>...: fail when this user changed anything under the directories (the system
# temporary directories) since <marker>, without crossing into other file systems (Agent Rule 7). find's
# errors are kept and read: "Permission denied" lines are normal in shared directories and are counted,
# while a directory that is not there, any other error, or a failure with no message fails the check, so
# a find that could not look is never read as a clean one. find prints each hit as "changed <path>".
check_root_fs() {
  local marker="$1" dir out line status=0 denied=0 leaked="" failed=""
  shift
  for dir in "$@"; do
    if [ ! -d "$dir" ]; then
      echo "ttsim: $dir is not a directory, so the root-filesystem check cannot run; nothing is swapped in" >&2
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
    echo "ttsim: the root-filesystem check over $* did not run cleanly (find rc=$status); nothing is swapped in" >&2
    printf '%s' "$failed" >&2
    return 1
  fi
  if [ -n "$leaked" ]; then
    echo "ttsim: files changed in the system temporary directories during the install; nothing is swapped in:" >&2
    printf '%s' "$leaked" >&2
    return 1
  fi
  echo "ttsim: root-filesystem check passed: nothing changed under $* since the start" \
    "(find rc=$status, $denied unreadable paths skipped)"
}

# swap_in: move a previous install at the prefix aside, never deleting it (its removal is an owner decision),
# then rename the staging prefix into place. When the rename fails, the previous install is put back. Sets
# aside to where the previous install went, or to "" when there was none, and made_staging to 0 once staging
# is in place.
swap_in() {
  aside=""
  if [ -e "$prefix" ] || [ -L "$prefix" ]; then
    aside="$prefix.previous-$(date +%Y%m%d-%H%M%S)"
    if [ -e "$aside" ] || [ -L "$aside" ]; then
      echo "ttsim: $aside already exists; nothing is swapped in" >&2
      return 1
    fi
    if ! mv -T "$prefix" "$aside"; then
      echo "ttsim: cannot move $prefix aside to $aside; nothing is swapped in" >&2
      return 1
    fi
    echo "ttsim: moved the previous install aside to $aside"
  fi
  if ! mv -T "$staging" "$prefix"; then
    if [ -n "$aside" ] && ! mv -T "$aside" "$prefix"; then
      echo "ttsim: cannot rename $staging to $prefix, nor put the previous install back; it stays at $aside" >&2
      return 1
    fi
    echo "ttsim: cannot rename $staging to $prefix; the previous install is back in place" >&2
    return 1
  fi
  made_staging=0
}

# Everything this run creates, and nothing else, is removed on exit: its start marker, and its staging prefix
# while that has not been swapped in.
marker=""
made_staging=0
cleanup() {
  if [ -n "$marker" ]; then rm -f "$marker"; fi
  if [ "$made_staging" = 1 ]; then rm -rf "$staging"; fi
}
trap cleanup EXIT

if [ -e "$staging" ] || [ -L "$staging" ]; then
  echo "ttsim: $staging exists from an earlier run that did not finish; inspect it and move it aside (never delete it), then rerun" >&2
  exit 1
fi
# The root-filesystem check below lists files changed after this marker.
marker="$(mktemp "$TMPDIR/ttsim-start.XXXXXX")"
check_space "$INSTALL_KIB"
descriptor="$(source_descriptor)"

downloads="$LASSI_SCRATCH/downloads/ttsim-$VERSION"
mkdir -p "$downloads"
library="$downloads/$LIBRARY"
fetch "$URL" "$SHA256" "$library"
size="$(stat -c %s "$library")"
if [ "$size" != "$SIZE" ]; then
  echo "ttsim: $library has $size bytes, not the pinned $SIZE" >&2
  exit 1
fi

mkdir -p "$LASSI_TOOLCHAINS"
mkdir "$staging"
made_staging=1
cp "$library" "$staging/$LIBRARY"
cp "$descriptor" "$staging/$SOC_DESCRIPTOR"
chmod 0644 "$staging/$LIBRARY" "$staging/$SOC_DESCRIPTOR"
pinned_record >"$staging/$record_name"
if ! installed_matches "$staging"; then
  echo "ttsim: the staged files in $staging do not match the pin" >&2
  exit 1
fi

# Nothing may land on the root filesystem (Agent Rule 7): refuse the swap when this user changed anything in
# the system temporary directories since the run started, or when that cannot be checked.
check_root_fs "$marker" /tmp /var/tmp

# Swap: a previous install is moved aside, never deleted, and staging is renamed into place.
aside=""
swap_in

if ! installed_matches "$prefix"; then
  echo "ttsim: $prefix does not match the pin after the swap" >&2
  exit 1
fi
echo "ttsim: installed $PREFIX_NAME: $LIBRARY with $SOC_DESCRIPTOR beside it"
if [ -n "$aside" ]; then
  echo "ttsim: the previous install stays at $aside until the owner decides to remove it"
fi
sha256sum "$prefix/$LIBRARY" "$prefix/$SOC_DESCRIPTOR"
