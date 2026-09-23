#!/usr/bin/env bash
# Install the pinned CUDA toolkit in user space (task P0.7), as recorded in toolchains/cuda.pin.
# Run on the build host through `uv run tools/rx.py job start --big`. Idempotent: an install whose
# nvcc already reports the pinned version is kept, and a finished download is reused.
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
# Vendor installers put temporary files under TMPDIR; keep them on the scratch disk (Agent Rule 7).
export TMPDIR="$LASSI_SCRATCH/tmp"
mkdir -p "$TMPDIR"
prefix="$LASSI_TOOLCHAINS/$PREFIX_NAME"
nvcc="$prefix/bin/nvcc"

if [ -x "$nvcc" ] && "$nvcc" --version | grep -qF "$EXPECT_VERSION"; then
  echo "cuda: $prefix already holds the pinned toolkit"
  "$nvcc" --version
  exit 0
fi

downloads="$LASSI_SCRATCH/downloads"
mkdir -p "$downloads"
runfile="$downloads/$(basename "$URL")"
if ! echo "$MD5  $runfile" | md5sum -c --status 2>/dev/null; then
  curl -fsSL --retry 3 -C - -o "$runfile" "$URL"
  echo "$MD5  $runfile" | md5sum -c -
fi

work="$TMPDIR/cuda-install.$$"
mkdir -p "$work"
# Without a writable /var/log the runfile installer logs to /tmp/cuda-installer.log on the root
# filesystem, a transient write disclosed in plans/OWNER-QUEUE.md (OQ-010). On every exit path except
# SIGKILL the log moves into the install prefix, or into scratch when the prefix does not exist.
clear_root_log() {
  local log=/tmp/cuda-installer.log
  if [ -f "$log" ] && [ -O "$log" ]; then
    if [ -d "$prefix" ]; then mv "$log" "$prefix/cuda-installer.log"; else mv "$log" "$TMPDIR/cuda-installer.log"; fi
  fi
}
trap clear_root_log EXIT
# INSTALL_FLAGS is a list of separate flags, so it is split on purpose.
# shellcheck disable=SC2086
sh "$runfile" $INSTALL_FLAGS --toolkitpath="$prefix" --tmpdir="$work"
rm -rf "$work"
clear_root_log

if ! "$nvcc" --version | grep -qF "$EXPECT_VERSION"; then
  echo "cuda: $nvcc does not report $EXPECT_VERSION" >&2
  exit 1
fi
echo "cuda: installed $PREFIX_NAME"
"$nvcc" --version
