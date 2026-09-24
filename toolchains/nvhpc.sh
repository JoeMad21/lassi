#!/usr/bin/env bash
# Install the pinned NVIDIA HPC SDK in user space (task P0.7), as recorded in toolchains/nvhpc.pin.
# Run on the build host through `uv run tools/rx.py job start --big`. Idempotent: an install whose
# nvc++ already reports the pinned version is kept, and a finished download is reused.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=nvhpc.pin
source "$here/nvhpc.pin"
: "${LASSI_TOOLCHAINS:?LASSI_TOOLCHAINS must be set; the gate sets it}"
: "${LASSI_SCRATCH:?LASSI_SCRATCH must be set; the gate sets it}"
for root in "$LASSI_TOOLCHAINS" "$LASSI_SCRATCH"; do
  case "$root" in
    /mnt/nvme10/*) ;;
    *) echo "nvhpc: refusing to work outside /mnt/nvme10 (Agent Rule 7): $root" >&2; exit 2 ;;
  esac
done
# Vendor installers put temporary files under TMPDIR; keep them on the scratch disk (Agent Rule 7).
export TMPDIR="$LASSI_SCRATCH/tmp"
mkdir -p "$TMPDIR"
prefix="$LASSI_TOOLCHAINS/$PREFIX_NAME"
nvcpp="$prefix/$COMPILER_SUBDIR/nvc++"

if [ -x "$nvcpp" ] && "$nvcpp" --version | grep -qF "$EXPECT_VERSION"; then
  echo "nvhpc: $prefix already holds the pinned SDK"
  "$nvcpp" --version
  exit 0
fi

downloads="$LASSI_SCRATCH/downloads"
mkdir -p "$downloads"
tarball="$downloads/$(basename "$URL")"
done_marker="$tarball.complete"
if [ ! -f "$done_marker" ]; then
  curl -fsSL --retry 3 -C - -o "$tarball" "$URL"
  touch "$done_marker"
fi
sum="$(sha256sum "$tarball" | cut -d' ' -f1)"
if [ "$SHA256" != "PLACEHOLDER" ] && [ "$sum" != "$SHA256" ]; then
  echo "nvhpc: $tarball has sha256 $sum, not the pinned $SHA256" >&2
  exit 1
fi
echo "nvhpc: tarball sha256 $sum"

src="$LASSI_SCRATCH/src/nvhpc-$VERSION"
rm -rf "$src"
mkdir -p "$src"
tar -xzf "$tarball" -C "$src" --strip-components=1
# INSTALL_FLAGS holds NAME=value settings for the installer's environment, so it is split on purpose.
# shellcheck disable=SC2086
(cd "$src" && env $INSTALL_FLAGS NVHPC_INSTALL_DIR="$prefix" ./install)
rm -rf "$src"

if ! "$nvcpp" --version | grep -qF "$EXPECT_VERSION"; then
  echo "nvhpc: $nvcpp does not report $EXPECT_VERSION" >&2
  exit 1
fi
echo "nvhpc: installed $PREFIX_NAME"
"$nvcpp" --version
