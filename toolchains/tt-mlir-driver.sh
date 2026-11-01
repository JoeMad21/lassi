#!/usr/bin/env bash
# Build lassi-ttmlir-run, the LASSI tt-mlir driver (task P17.48), as recorded in toolchains/tt-mlir-driver.pin, against
# the pinned runtime install of toolchains/tt-mlir-runtime.pin. On alpha01 it is a short CPU-only run:
#   uv run tools/rx.py run --timeout 600 -- 'LASSI_ALLOWED_ROOT=/mnt/nvme10/joseph_ufl bash toolchains/tt-mlir-driver.sh'
# Under $LASSI_TOOLCHAINS/$PREFIX_NAME it places bin/lassi-ttmlir-run and the install record lassi-install.txt; the
# ttsim system descriptor is written there afterwards, in the sandbox, by tools/ttmlir_system_desc.py. It compiles one
# C++ file with the runtime's own compiler and linker and lists the program's libraries with ldd (the loader's trace
# mode, which runs none of their code); it runs nothing it builds and opens no device. Idempotent: an install whose
# record matches the pin is kept. Agents never delete on the build host, so anything else at the prefix that holds a
# record is refused and left as it is; an unfinished build of the same pin is built over (its files are written
# again, never removed).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/.." && pwd)"
# shellcheck source=tt-mlir-driver.pin
source "$here/tt-mlir-driver.pin"
set_by="on alpha01 by the gate (LASSI_TOOLCHAINS, LASSI_SCRATCH) or the rx command line (LASSI_ALLOWED_ROOT)"
: "${LASSI_TOOLCHAINS:?LASSI_TOOLCHAINS must be set $set_by}"
: "${LASSI_SCRATCH:?LASSI_SCRATCH must be set $set_by}"
: "${LASSI_ALLOWED_ROOT:?LASSI_ALLOWED_ROOT must be set $set_by}"
fail() {
  echo "tt-mlir-driver: $*" >&2
  exit 2
}
case "$LASSI_ALLOWED_ROOT" in
  / | [!/]*) fail "LASSI_ALLOWED_ROOT must be an absolute path other than /, not '$LASSI_ALLOWED_ROOT'" ;;
esac
for root in "$LASSI_TOOLCHAINS" "$LASSI_SCRATCH"; do
  case "$(realpath -m "$root")/" in
    "$LASSI_ALLOWED_ROOT"/*) ;;
    *) fail "refusing to work outside $LASSI_ALLOWED_ROOT (Agent Rule 7): $root" ;;
  esac
done
for name in SOURCE_SHA256 HEADER_SHA256 VERSION PREFIX_NAME; do
  case "${!name}" in
    *PLACEHOLDER*) fail "toolchains/tt-mlir-driver.pin's $name is a PLACEHOLDER; fill it from the source first" ;;
  esac
done
# A tool that crashes must leave no core (Agent Rule 7): exactly 1 byte, soft and hard, as the other install scripts
# hold it, read back by prlimit.
prlimit --pid $$ --core=1:1 || fail "cannot set RLIMIT_CORE to exactly 1 byte (Agent Rule 7)"
[ "$(prlimit --pid $$ --core --output SOFT,HARD --noheadings | tr -s ' ' | sed 's/^ //')" = "1 1" ] \
  || fail "RLIMIT_CORE does not read back as 1 1 (Agent Rule 7)"
export TMPDIR="$LASSI_SCRATCH/tmp"
mkdir -p "$TMPDIR"
unset CC CXX CFLAGS CXXFLAGS CPPFLAGS LDFLAGS CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH \
  LD_LIBRARY_PATH LD_RUN_PATH COMPILER_PATH
# The pinned values come back from the pin file itself, since a sourced pin sets CXX and CXXFLAGS too.
pin_cxx="$(source "$here/tt-mlir-driver.pin" && printf '%s' "$CXX")"
pin_cxxflags="$(source "$here/tt-mlir-driver.pin" && printf '%s' "$CXXFLAGS")"
# The runtime pin's values, read in a subshell so that its NAME and VERSION do not replace this pin's.
runtime() { (source "$here/$RUNTIME_PIN.pin" && eval "printf '%s' \"\${$1}\""); }
runtime_prefix="$LASSI_TOOLCHAINS/$(runtime PREFIX_NAME)"
runtime_head="$(runtime NAME) $(runtime COMMIT) $(runtime URL) $(runtime VERSION) $(runtime BUILD)"
runtime_library="$runtime_prefix/$(runtime RUNTIME_LIBRARY)"
metal_libraries="$runtime_prefix/$(runtime METAL_BUILD_DIR)/lib"
[ "$(head -n 1 "$runtime_prefix/lassi-install.txt" 2>/dev/null || true)" = "$runtime_head" ] \
  || fail "$runtime_prefix is not the pinned runtime install: its lassi-install.txt does not start with" \
    "'$runtime_head'; install it with toolchains/$RUNTIME_PIN.sh"
[ -f "$runtime_library" ] || fail "the runtime library $runtime_library is missing"
# The source and the header must be the pinned bytes, and VERSION must name them.
source_file="$here/$SOURCE"
header_file="$repo/$HEADER"
digest() { sha256sum "$1" | cut -d ' ' -f 1; }
[ "$(digest "$source_file")" = "$SOURCE_SHA256" ] || fail "$source_file is not the pinned source (SOURCE_SHA256)"
[ "$(digest "$header_file")" = "$HEADER_SHA256" ] || fail "$header_file is not the pinned header (HEADER_SHA256)"
[ "$(cat "$source_file" "$header_file" | sha256sum | cut -c 1-12)" = "$VERSION" ] \
  || fail "VERSION is not the first 12 hex digits of the sha256 of the source followed by the header"
"$pin_cxx" --version | grep -F "$CXX_EXPECT" >/dev/null || fail "$pin_cxx does not report '$CXX_EXPECT'"
prefix="$LASSI_TOOLCHAINS/$PREFIX_NAME"
record() {
  echo "$NAME $VERSION $SOURCE_SHA256 $HEADER_SHA256 $(runtime PREFIX_NAME)"
  echo "compiler $pin_cxx ($CXX_EXPECT); linker $LINKER; flags $pin_cxxflags"
}
if [ -f "$prefix/lassi-install.txt" ]; then
  [ "$(cat "$prefix/lassi-install.txt")" = "$(record)" ] \
    || fail "$prefix holds another install record; it is left as it is"
  echo "tt-mlir-driver: $prefix is already installed at this pin"
  exit 0
fi
includes=()
for dir in $INCLUDE_DIRS; do
  [ -d "$runtime_prefix/$dir" ] || fail "the include directory $runtime_prefix/$dir is missing"
  includes+=("-I$runtime_prefix/$dir")
done
mkdir -p "$prefix/bin"
partial="$prefix/$BINARY.partial"
# shellcheck disable=SC2086
"$pin_cxx" $pin_cxxflags --ld-path="$LINKER" "-I$(dirname "$header_file")" "${includes[@]}" "$source_file" \
  -o "$partial" "-L$(dirname "$runtime_library")" -lTTMLIRRuntime \
  "-Wl,-rpath,$(dirname "$runtime_library")" "-Wl,-rpath-link,$metal_libraries" -Wl,--allow-shlib-undefined
missing="$(ldd "$partial" | grep -c 'not found' || true)"
[ "$missing" = "0" ] || { ldd "$partial" >&2; fail "the driver has $missing libraries the loader cannot find"; }
mv -T "$partial" "$prefix/$BINARY"
record >"$prefix/lassi-install.unfinished"
mv -T "$prefix/lassi-install.unfinished" "$prefix/lassi-install.txt"
echo "tt-mlir-driver: installed $prefix/$BINARY"
sha256sum "$prefix/$BINARY"
echo "tt-mlir-driver: next, write the ttsim system descriptor with tools/ttmlir_system_desc.py"
