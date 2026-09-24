#!/usr/bin/env bash
# Install the pinned CUDA toolkit in user space (tasks P0.7 and P0.19), as recorded in toolchains/cuda.pin.
# The toolkit is assembled from NVIDIA's per-component redistributable archives. The redistrib manifest is
# checked against its pinned sha256, the pin against the manifest, and every archive against the manifest's
# sha256 before it is extracted; no runfile or other installer runs (OQ-010). Run on the build host through
# `uv run tools/rx.py job start --big`.
# Idempotent: an install whose nvcc reports the pinned version and whose component record matches the pin is
# kept, and verified downloads are reused. A new install is assembled in a staging prefix and checked with
# sm_80 builds; only then is a previous install moved aside (never deleted) and the staging prefix renamed
# into place.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=cuda.pin
source "$here/cuda.pin"
: "${LASSI_TOOLCHAINS:?LASSI_TOOLCHAINS must be set; the gate sets it}"
: "${LASSI_SCRATCH:?LASSI_SCRATCH must be set; the gate sets it}"
for root in "$LASSI_TOOLCHAINS" "$LASSI_SCRATCH"; do
  case "$root" in
    /mnt/nvme10/*) ;;
    *) echo "cuda: refusing to work outside /mnt/nvme10 (Agent Rule 7): $root" >&2; exit 2 ;;
  esac
done
# Compilers, and bash for here-documents, put temporary files under TMPDIR; keep them on the scratch disk
# (Agent Rule 7).
export TMPDIR="$LASSI_SCRATCH/tmp"
mkdir -p "$TMPDIR"
prefix="$LASSI_TOOLCHAINS/$PREFIX_NAME"
nvcc="$prefix/bin/nvcc"
staging="$prefix.staging"
# The record an install from this script carries, naming the manifest, the archives, and the layout it used.
record_name=redist-components.txt

# archive_spec <component>: print the pin's "<version> <size> <sha256> <relative path>" for the component.
archive_spec() {
  local key="ARCHIVE_${1^^}"
  if [ -z "${!key:-}" ]; then
    echo "cuda: cuda.pin has no $key for the component $1" >&2
    return 1
  fi
  printf '%s\n' "${!key}"
}

# pinned_record: print the component record an install from the current pin carries.
pinned_record() {
  local c
  echo "manifest $MANIFEST_URL $MANIFEST_SHA256"
  for c in $COMPONENTS; do
    echo "$c $(archive_spec "$c")"
  done
  echo "layout $LAYOUT_PREFIX_DIRS; $LAYOUT_TARGET_DIR: $LAYOUT_TARGET_DIRS; $LAYOUT_LINKS; skip $LAYOUT_SKIP"
}

# reports_pinned_version <nvcc>: succeed when `<nvcc> --version` prints EXPECT_VERSION.
reports_pinned_version() {
  local out
  out="$("$1" --version 2>&1)" || return 1
  grep -qF "$EXPECT_VERSION" <<<"$out"
}

if [ -x "$nvcc" ] && reports_pinned_version "$nvcc" && [ -f "$prefix/$record_name" ] \
  && [ "$(cat "$prefix/$record_name")" = "$(pinned_record)" ]; then
  echo "cuda: $prefix already holds the pinned toolkit"
  "$nvcc" --version
  exit 0
fi

# check_space <planned KiB>: refuse when the planned bytes do not fit in the free space of the scratch disk,
# or when, added to what the scratch root holds now (du -s), they would pass the owner's cap of 120 GiB on
# the scratch root (plans/p0-core.md, P0.19).
check_space() {
  local planned="$1" cap=$((120 * 1024 * 1024)) used free value
  used="$( (du -sk "$LASSI_SCRATCH" 2>/dev/null || true) | cut -f1)"
  free="$(df -Pk "$LASSI_SCRATCH" | awk 'NR == 2 { print $4 }')"
  for value in "$used" "$free"; do
    case "$value" in
      '' | *[!0-9]*)
        echo "cuda: cannot measure the scratch disk (du -s gave '$used' KiB, df gave '$free' KiB free)" >&2
        return 1
        ;;
    esac
  done
  if [ $((used + planned)) -gt "$cap" ]; then
    echo "cuda: $LASSI_SCRATCH holds $used KiB; the planned $planned KiB would pass the 120 GiB cap ($cap KiB)" >&2
    echo "cuda: free space through the owner queue before installing" >&2
    return 1
  fi
  if [ "$planned" -gt "$free" ]; then
    echo "cuda: the planned $planned KiB do not fit in the $free KiB free on the scratch disk" >&2
    return 1
  fi
  echo "cuda: space check passed: $used KiB used under $LASSI_SCRATCH, $planned KiB planned, cap $cap KiB"
}

# fetch <url> <sha256> <file>: keep <file> when it already has the sha256; otherwise download it to a new
# temporary name next to it, check the sha256, and rename it into place. A file that exists with another
# sha256 is refused and left as it is, since this run did not create it.
fetch() {
  local url="$1" sum="$2" file="$3" part
  if [ -e "$file" ]; then
    if echo "$sum  $file" | sha256sum -c --status; then
      echo "cuda: reusing $file (sha256 $sum)"
      return 0
    fi
    echo "cuda: $file exists but its sha256 is not $sum; inspect it, then rerun" >&2
    return 1
  fi
  part="$(mktemp "$file.part.XXXXXX")"
  if ! curl -fsSL --retry 3 -o "$part" "$url"; then
    rm -f "$part"
    echo "cuda: cannot download $url" >&2
    return 1
  fi
  if ! echo "$sum  $part" | sha256sum -c --status; then
    rm -f "$part"
    echo "cuda: $url does not have the sha256 $sum" >&2
    return 1
  fi
  mv "$part" "$file"
  echo "cuda: downloaded $file (sha256 $sum)"
}

# manifest_entry <component>: print the manifest's "<version> <size> <sha256> <relative path>" for the
# component's linux-x86_64 archive, in the order of the pin's ARCHIVE_ values.
manifest_entry() {
  python3 - "$manifest" "$1" <<'EOF'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    entry = json.load(handle)[sys.argv[2]]
archive = entry["linux-x86_64"]
print(entry["version"], archive["size"], archive["sha256"], archive["relative_path"])
EOF
}

# layout_dest <component> <entry>: print the directory a top-level archive entry goes to, by the pinned layout.
layout_dest() {
  local dir
  for dir in $LAYOUT_PREFIX_DIRS; do
    if [ "$2" = "$dir" ]; then echo "$staging"; return 0; fi
  done
  for dir in $LAYOUT_TARGET_DIRS; do
    if [ "$2" = "$dir" ]; then echo "$staging/$LAYOUT_TARGET_DIR"; return 0; fi
  done
  if [ "$2" = LICENSE ]; then echo "$staging/licenses/$1"; return 0; fi
  return 1
}

# layout_skips <entry>: succeed when the pin leaves a top-level archive entry out of the install (LAYOUT_SKIP).
layout_skips() {
  local name
  for name in $LAYOUT_SKIP; do
    if [ "$1" = "$name" ]; then return 0; fi
  done
  return 1
}

# extract_component <component> <archive>: extract the archive's top-level entries where the layout puts
# them. Before anything is extracted, every member must sit under the archive's own top directory and every
# top-level entry must have a place in the layout or be one it leaves out; tar -k then refuses to replace a
# file another component already placed.
extract_component() {
  local component="$1" archive="$2" top members member entries entry dest
  top="$(basename "$archive" .tar.xz)"
  members="$(tar -tJf "$archive")"
  while IFS= read -r member; do
    case "$member" in
      "$top"/*) ;;
      *)
        echo "cuda: $archive holds $member, outside its top directory $top" >&2
        return 1
        ;;
    esac
  done <<<"$members"
  entries="$(cut -d/ -f2 <<<"$members" | sort -u)"
  for entry in $entries; do
    if layout_skips "$entry"; then continue; fi
    if ! layout_dest "$component" "$entry" >/dev/null; then
      echo "cuda: $archive holds $entry, which the layout in cuda.pin does not place" >&2
      return 1
    fi
  done
  for entry in $entries; do
    if layout_skips "$entry"; then
      echo "cuda: leaving $component's $entry out of the install (LAYOUT_SKIP in cuda.pin)"
      continue
    fi
    dest="$(layout_dest "$component" "$entry")"
    mkdir -p "$dest"
    tar -xJf "$archive" -C "$dest" --strip-components=1 --keep-old-files "$top/$entry"
  done
}

# check_compiles <cuda prefix>: repeat the P0.7 compile checks (plans/spikes/p0-toolchains-verify.md) with
# the bible's LASSI flags: a kernel for sm_80 with the prefix's nvcc, and an OpenMP target loop for cc80 with
# the pinned nvc++ and NVHPC_CUDA_HOME at the prefix. Each binary must hold an sm_80 cubin; none is run. The
# compilers get the check directory as TMPDIR, since nvc++ leaves a small nvacc*.bc file there per build.
check_compiles() {
  local home="$1" nvcpp
  check="$(mktemp -d "$TMPDIR/cuda-check.XXXXXX")"
  cat >"$check/t.cu" <<'EOF'
#include <cstdio>
__global__ void add(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}
int main() { std::printf("built\n"); return 0; }
EOF
  (cd "$check" && TMPDIR="$check" LC_ALL=C "$home/bin/nvcc" -std=c++14 -Xcompiler -Wall -arch=sm_80 -O3 -o t t.cu)
  "$home/bin/cuobjdump" --list-elf "$check/t" | grep -F .sm_80.cubin
  echo "cuda: nvcc built an sm_80 cubin"
  # The pinned nvc++ path, from nvhpc.pin; sourced in a subshell so its keys do not replace this pin's.
  nvcpp="$LASSI_TOOLCHAINS/$(source "$here/nvhpc.pin" && echo "$PREFIX_NAME/$COMPILER_SUBDIR")/nvc++"
  if [ ! -x "$nvcpp" ]; then
    echo "cuda: $nvcpp is not installed, so the nvc++ check is skipped"
    return 0
  fi
  cat >"$check/t.cpp" <<'EOF'
int main() {
  const int n = 1024; float x[n], y[n];
  for (int i = 0; i < n; i++) { x[i] = 1.0f; y[i] = 2.0f; }
  #pragma omp target teams distribute parallel for map(to: x[:n]) map(tofrom: y[:n])
  for (int i = 0; i < n; i++) y[i] += 3.0f * x[i];
  return y[0] > 4.0f ? 0 : 1;
}
EOF
  (cd "$check" && TMPDIR="$check" LC_ALL=C NVHPC_CUDA_HOME="$home" "$nvcpp" -Wall -O3 -Minfo -mp=gpu -gpu=cc80 -o tp t.cpp)
  "$home/bin/cuobjdump" --list-elf "$check/tp" | grep -F .sm_80.cubin
  echo "cuda: nvc++ built an sm_80 cubin with NVHPC_CUDA_HOME at $home"
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
      echo "cuda: $dir is not a directory, so the root-filesystem check cannot run; nothing is swapped in" >&2
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
    echo "cuda: the root-filesystem check over $* did not run cleanly (find rc=$status); nothing is swapped in" >&2
    printf '%s' "$failed" >&2
    return 1
  fi
  if [ -n "$leaked" ]; then
    echo "cuda: files changed in the system temporary directories during the install; nothing is swapped in:" >&2
    printf '%s' "$leaked" >&2
    return 1
  fi
  echo "cuda: root-filesystem check passed: nothing changed under $* since the start" \
    "(find rc=$status, $denied unreadable paths skipped)"
}

# swap_in: move a previous install at the prefix aside, never deleting it (its removal is an owner
# decision), then rename the staging prefix into place. An install without the component record is the
# P0.7 runfile install. When the rename fails, the previous install is put back. Sets aside to where the
# previous install went, or to "" when there was none, and made_staging to 0 once staging is in place.
swap_in() {
  local kind
  aside=""
  if [ -e "$prefix" ] || [ -L "$prefix" ]; then
    kind=runfile
    if [ -f "$prefix/$record_name" ]; then kind=redist; fi
    aside="$prefix.$kind-$(date +%Y%m%d-%H%M%S)"
    if [ -e "$aside" ] || [ -L "$aside" ]; then
      echo "cuda: $aside already exists; nothing is swapped in" >&2
      return 1
    fi
    if ! mv -T "$prefix" "$aside"; then
      echo "cuda: cannot move $prefix aside to $aside; nothing is swapped in" >&2
      return 1
    fi
    echo "cuda: moved the previous install aside to $aside"
  fi
  if ! mv -T "$staging" "$prefix"; then
    if [ -n "$aside" ] && ! mv -T "$aside" "$prefix"; then
      echo "cuda: cannot rename $staging to $prefix, nor put the previous install back; it stays at $aside" >&2
      return 1
    fi
    echo "cuda: cannot rename $staging to $prefix; the previous install is back in place" >&2
    return 1
  fi
  made_staging=0
}

# Everything this run creates, and nothing else, is removed on exit: its check directory, its start marker,
# and its staging prefix while that has not been swapped in.
check=""
marker=""
made_staging=0
cleanup() {
  if [ -n "$check" ]; then rm -rf "$check"; fi
  if [ -n "$marker" ]; then rm -f "$marker"; fi
  if [ "$made_staging" = 1 ]; then rm -rf "$staging"; fi
}
trap cleanup EXIT

if [ -e "$staging" ] || [ -L "$staging" ]; then
  echo "cuda: $staging exists from an earlier run that did not finish; inspect it and remove it, then rerun" >&2
  exit 1
fi
# The root-filesystem check below lists files changed after this marker.
marker="$(mktemp "$TMPDIR/cuda-start.XXXXXX")"

planned_kib=$INSTALL_KIB
for c in $COMPONENTS; do
  spec="$(archive_spec "$c")"
  read -r _ size _ _ <<<"$spec"
  planned_kib=$((planned_kib + size / 1024 + 1))
done
check_space "$planned_kib"

downloads="$LASSI_SCRATCH/downloads"
mkdir -p "$downloads"
manifest="$downloads/$(basename "$MANIFEST_URL")"
fetch "$MANIFEST_URL" "$MANIFEST_SHA256" "$manifest"

mkdir "$staging"
made_staging=1
mkdir -p "$staging/$LAYOUT_TARGET_DIR"
for c in $COMPONENTS; do
  spec="$(archive_spec "$c")"
  entry="$(manifest_entry "$c")"
  if [ "$entry" != "$spec" ]; then
    echo "cuda: cuda.pin has ARCHIVE_${c^^}=\"$spec\", but the manifest has \"$entry\"" >&2
    exit 1
  fi
  read -r version size sum path <<<"$entry"
  archive="$downloads/$(basename "$path")"
  fetch "$URL/$path" "$sum" "$archive"
  extract_component "$c" "$archive"
  echo "cuda: placed $c $version"
done
for link in $LAYOUT_LINKS; do
  ln -s "${link#*:}" "$staging/${link%%:*}"
done
pinned_record >"$staging/$record_name"

if ! reports_pinned_version "$staging/bin/nvcc"; then
  echo "cuda: $staging/bin/nvcc does not report $EXPECT_VERSION" >&2
  exit 1
fi
check_compiles "$staging"

# Nothing may land on the root filesystem (Agent Rule 7): refuse the swap when this user changed anything
# in the system temporary directories since the run started, or when that cannot be checked.
check_root_fs "$marker" /tmp /var/tmp

# Swap: a previous install is moved aside, never deleted, and staging is renamed into place.
aside=""
swap_in

if ! reports_pinned_version "$nvcc"; then
  echo "cuda: $nvcc does not report $EXPECT_VERSION after the swap" >&2
  exit 1
fi
echo "cuda: installed $PREFIX_NAME from the redistributable archives"
if [ -n "$aside" ]; then
  echo "cuda: the previous install stays at $aside until the owner decides to remove it"
fi
du -sk "$prefix"
"$nvcc" --version
