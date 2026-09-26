#!/usr/bin/env bash
# Build the pinned tt-metal in user space (task P4.2), as recorded in toolchains/tt-metal.pin. Run on the build
# host through `uv run tools/rx.py job start --big`. It fetches the pinned commit and its submodules by sha at
# depth 1, configures with the pin's flags, and builds only the pin's TARGETS and examples: the runtime library,
# the linker scripts and objects the kernel JIT links with, and the pinned programming examples, not all or the
# install target, so that it takes the least CPU on the shared host (owner, 2026-09-25). It never runs a program
# it builds, so nothing opens a device.
# The flags are those of the pin's own ttsim CI build (.github/workflows/ttsim.yaml, pr-gate.yaml, and
# build-artifact.yaml at COMMIT: build_metal.sh --build-type Release with TOOLCHAIN_FILE, --disable-profiler,
# and --without-distributed), except the Python bindings, the tests, and ccache, which are off. At the pin none
# of those changes a compile flag of the targets built here: the bindings option only adds TT-NN code (outside
# TT-NN it prints a message, cmake/project_options.cmake:27), the test options only add test directories
# (CMakeLists.txt:272, tools/CMakeLists.txt:6, tt_metal/CMakeLists.txt:318, tt_stl/CMakeLists.txt:59), and ccache
# only caches. So the examples here come from the same sources and flags as that build's tt-metalium, though CI
# built them apart, with gcc-12 against its installed packages (ttsim.yaml), and here they build in the tree
# with the pin's clang-20 toolchain. Each links only TT::Metalium, with TT::STL (programming_examples/
# CMakeLists.txt:2) and, for the matmul ones, the header library Matmul::Common (matmul/matmul_common/
# CMakeLists.txt:1). UMD's build-time lint is already off (tt_metal/third_party/CMakeLists.txt:4). tt_metal runs
# python3 from PATH to generate its inspector RPC code (tt_metal/impl/CMakeLists.txt:145 and 218), so the script
# prints which one.
# The tree is built in place at $LASSI_TOOLCHAINS/$PREFIX_NAME: a CMake build tree holds absolute paths (its
# cache, its CPM sources, the run paths of the examples), so it cannot be built elsewhere and renamed into
# place. Instead the tree carries lassi-install.unfinished until every check has passed, and only then is that
# file renamed to the install record, lassi-install.txt. Idempotent: an install whose record matches the pin and
# whose tree checks out is kept; anything else at the prefix is moved aside, never deleted. A rerun at the same
# pin repeats the configure and lets Ninja continue a build that stopped; it does not repair an interrupted git
# checkout, which verify_tree refuses, or an interrupted CPM download, which CPM reuses as it finds it, so the
# configure or build fails or lassi-cpm-sources.txt shows changed files. For either, move the tree aside (never
# delete it) and rerun. A commit, submodule, sfpi, host tool, or CMake setting that differs from the pin is
# refused.
set -euo pipefail
# A tool that crashes leaves no core file, which the host's core handler would put on the root filesystem
# (Agent Rule 7).
ulimit -c 0
here="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=tt-metal.pin
source "$here/tt-metal.pin"
: "${LASSI_TOOLCHAINS:?LASSI_TOOLCHAINS must be set; the gate sets it}"
: "${LASSI_SCRATCH:?LASSI_SCRATCH must be set; the gate sets it}"
: "${LASSI_JOBS:?LASSI_JOBS must be set; the gate sets it}"
for root in "$LASSI_TOOLCHAINS" "$LASSI_SCRATCH"; do
  case "$(realpath -m "$root")/" in
    /mnt/nvme10/joseph_ufl/*) ;;
    *) echo "tt-metal: refusing to work outside /mnt/nvme10/joseph_ufl (Agent Rule 7): $root" >&2; exit 2 ;;
  esac
done
case "$LASSI_JOBS" in
  '' | *[!0-9]* | 0) echo "tt-metal: LASSI_JOBS must be a positive integer, not '$LASSI_JOBS'" >&2; exit 2 ;;
esac
prefix="$LASSI_TOOLCHAINS/$PREFIX_NAME"

# refuse_foreign <path>...: refuse when a path, with its links resolved, is FOREIGN_CHECKOUT or lies inside
# it; that checkout belongs to another project and is only ever read (plans/p4-ttsim.md).
refuse_foreign() {
  local path foreign
  foreign="$(realpath -m "$FOREIGN_CHECKOUT")"
  for path in "$@"; do
    case "$(realpath -m "$path")/" in
      "$foreign"/*)
        echo "tt-metal: refusing to write in $FOREIGN_CHECKOUT, another project's checkout: $path" >&2
        return 1
        ;;
    esac
  done
}

refuse_foreign "$prefix" "$LASSI_SCRATCH/tmp"
# Compilers, git, CMake, and bash for here-documents put temporary files under TMPDIR; keep them on the scratch
# disk (Agent Rule 7).
export TMPDIR="$LASSI_SCRATCH/tmp"
mkdir -p "$TMPDIR"
# The host environment adds nothing to the pinned configuration: no compiler, flag, search-path, or CPM
# variables, and no tt-metal runtime variables, which could name another tree (such as FOREIGN_CHECKOUT).
unset CC CXX CFLAGS CXXFLAGS CPPFLAGS LDFLAGS CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH \
  LD_LIBRARY_PATH CMAKE_PREFIX_PATH PKG_CONFIG_PATH CPM_USE_LOCAL_PACKAGES CPM_LOCAL_PACKAGES_ONLY
for name in $(compgen -e); do
  case "$name" in
    TT_METAL_* | TT_MLIR_* | ARCH_NAME) unset "$name" ;;
  esac
done
build="$prefix/$BUILD_DIR"
# The CPM source cache stays in the tree, tt-metal's own default, since the build refers to its sources by path.
cpm_cache="$prefix/.cpmcache"
record_name=lassi-install.txt
unfinished_name=lassi-install.unfinished

# git_in <dir> <args>...: run git in the repository whose .git is in <dir>, never in one found above it, with
# automatic maintenance off so that no background git outlives this script.
git_in() {
  local dir="$1"
  shift
  git --git-dir="$dir/.git" --work-tree="$dir" -C "$dir" -c gc.auto=0 -c maintenance.auto=false \
    -c advice.detachedHead=false "$@"
}

# pinned_record: print the record an install from the current pin carries, with the settings this script adds.
pinned_record() {
  local entry
  echo "tt-metal $COMMIT $URL"
  for entry in $SUBMODULES; do
    echo "submodule ${entry%%:*} ${entry##*:}"
  done
  echo "sfpi $SFPI_VERSION $SFPI_URL $SFPI_SHA256"
  echo "cpm $CPM_URL $CPM_SHA256"
  echo "host $CMAKE ($CMAKE_EXPECT); $NINJA ($NINJA_EXPECT); $CLANG_C, $CLANG_CXX ($CLANG_EXPECT);" \
    "$LINKER ($LINKER_EXPECT)"
  echo "toolchain $TOOLCHAIN_FILE"
  echo "flags $CMAKE_FLAGS"
  echo "local $(local_settings | tr '\n' ' ')"
  echo "build $BUILD_DIR; targets $TARGETS; examples $EXAMPLES"
}

# local_settings: print the -D settings this script adds to the pin's flags, one per line: the toolchain file,
# Ninja, and the CPM cache, which place the build, and no reading of the user or system CMake package registry
# (such as ~/.cmake/packages), which could hand the configure another project's packages. Nothing is
# installed, so no install prefix is set.
local_settings() {
  echo "-DCMAKE_TOOLCHAIN_FILE=$prefix/$TOOLCHAIN_FILE"
  echo "-DCMAKE_MAKE_PROGRAM=$NINJA"
  echo "-DCPM_SOURCE_CACHE=$cpm_cache"
  echo "-DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF"
  echo "-DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF"
}

# expect_version <expected> <command>...: refuse unless the first line the command prints holds <expected>.
expect_version() {
  local expected="$1" out
  shift
  if ! out="$("$@" 2>&1)"; then
    echo "tt-metal: cannot run $* to read its version" >&2
    return 1
  fi
  out="${out%%$'\n'*}"
  if [[ "$out" != *"$expected"* ]]; then
    echo "tt-metal: $1 reports '$out', not the pinned '$expected'; a pin change is a Decision Log entry" >&2
    return 1
  fi
  echo "tt-metal: $1: $out"
}

# check_host: refuse when a host build tool is not the one the pin names (Agent Rule 10): CMake and Ninja by
# path, and the compilers and the linker the toolchain file takes from PATH (ld.mold first); check_linked later
# reads which linker and compiler made the examples. It also prints the tools the build runs from PATH without a
# pin: python3, which tt_metal needs to generate code, and clang-format, if there is one.
check_host() {
  local tool
  expect_version "$CMAKE_EXPECT" "$CMAKE" --version
  expect_version "$NINJA_EXPECT" "$NINJA" --version
  for tool in "$CLANG_C" "$CLANG_CXX" "$LINKER"; do
    if ! command -v "$tool" >/dev/null; then
      echo "tt-metal: $tool, which the toolchain file takes from PATH, is not there" >&2
      return 1
    fi
  done
  expect_version "$CLANG_EXPECT" "$(command -v "$CLANG_C")" --version
  expect_version "$CLANG_EXPECT" "$(command -v "$CLANG_CXX")" --version
  expect_version "$LINKER_EXPECT" "$(command -v "$LINKER")" --version
  if command -v ld.mold >/dev/null; then
    echo "tt-metal: ld.mold is on PATH, so the toolchain file would link with it instead of $LINKER" >&2
    return 1
  fi
  if ! command -v python3 >/dev/null; then
    echo "tt-metal: python3 is not on PATH; tt_metal runs it to generate its inspector RPC code" >&2
    return 1
  fi
  echo "tt-metal: python3 $(command -v python3): $(python3 --version 2>&1)"
  if command -v clang-format >/dev/null; then
    echo "tt-metal: clang-format $(command -v clang-format): $(clang-format --version 2>&1 | head -n 1)"
  else
    echo "tt-metal: clang-format: not on PATH"
  fi
  echo "tt-metal: $(git --version)"
}

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
        echo "tt-metal: cannot measure the scratch disk (du -s gave '$used' KiB, df gave '$free' KiB free)" >&2
        return 1
        ;;
    esac
  done
  if [ $((used + planned)) -gt "$stop" ]; then
    echo "tt-metal: $LASSI_SCRATCH holds $used KiB; the planned $planned KiB would pass the 115 GiB stop line" \
      "($stop KiB)" >&2
    echo "tt-metal: queue backup or deletion candidates in the owner queue before installing" >&2
    return 1
  fi
  if [ "$planned" -gt "$free" ]; then
    echo "tt-metal: the planned $planned KiB do not fit in the $free KiB free on the scratch disk" >&2
    return 1
  fi
  echo "tt-metal: space check passed: $used KiB used under $LASSI_SCRATCH, $planned KiB planned, stop line $stop KiB"
}

# fetch_at <dir> <url> <commit>: make <dir> a repository whose HEAD is <commit>, fetched from <url> by sha at
# depth 1 with no tags. A repository already at <commit> is kept (a resumed run); one with another origin, or
# at another commit, is refused and left as it is.
fetch_at() {
  local dir="$1" url="$2" commit="$3" origin head
  mkdir -p "$dir"
  if [ ! -e "$dir/.git" ]; then
    git -c init.defaultBranch=main -C "$dir" init -q
  fi
  origin="$(git_in "$dir" remote get-url origin 2>/dev/null || true)"
  if [ -z "$origin" ]; then
    git_in "$dir" remote add origin "$url"
  elif [ "$origin" != "$url" ]; then
    echo "tt-metal: $dir fetches from $origin, not $url; move $prefix aside (never delete it) and rerun" >&2
    return 1
  fi
  head="$(git_in "$dir" rev-parse -q --verify 'HEAD^{commit}' || true)"
  if [ "$head" = "$commit" ]; then
    echo "tt-metal: $dir is already at $commit"
    return 0
  fi
  if [ -n "$head" ]; then
    echo "tt-metal: $dir is at $head, not $commit; move $prefix aside (never delete it) and rerun" >&2
    return 1
  fi
  git_in "$dir" fetch -q --depth 1 --no-tags origin "$commit"
  git_in "$dir" checkout -q --detach "$commit"
  head="$(git_in "$dir" rev-parse HEAD)"
  if [ "$head" != "$commit" ]; then
    echo "tt-metal: $dir checked out $head, not the pinned $commit" >&2
    return 1
  fi
  echo "tt-metal: fetched $url at $commit"
}

# fetch_submodules: refuse unless the tree at COMMIT holds exactly the pinned gitlinks, then fetch each
# submodule at its pinned commit from the URL the tree's .gitmodules gives it.
fetch_submodules() {
  local links pinned entry path url
  links="$(git_in "$prefix" ls-tree -r HEAD | awk '$2 == "commit" { print $4 ":" $3 }' | LC_ALL=C sort)"
  pinned="$(tr ' ' '\n' <<<"$SUBMODULES" | LC_ALL=C sort)"
  if [ "$links" != "$pinned" ]; then
    echo "tt-metal: the gitlinks at $COMMIT are not the pinned SUBMODULES" >&2
    printf 'in the tree:\n%s\nin the pin:\n%s\n' "$links" "$pinned" >&2
    return 1
  fi
  for entry in $SUBMODULES; do
    path="${entry%%:*}"
    url="$(git_in "$prefix" config -f .gitmodules --get "submodule.$path.url")"
    fetch_at "$prefix/$path" "$url" "${entry##*:}"
  done
}

# verify_tree: refuse unless the tree and every submodule are at their pinned commits, no tracked file differs
# from them, and tt_metal/sfpi-version names the pinned sfpi and its sha256.
verify_tree() {
  local entry path head changed
  head="$(git_in "$prefix" rev-parse HEAD || true)"
  if [ "$head" != "$COMMIT" ]; then
    echo "tt-metal: $prefix is at '$head', not the pinned $COMMIT" >&2
    return 1
  fi
  for entry in $SUBMODULES; do
    path="${entry%%:*}"
    head=""
    if [ -e "$prefix/$path/.git" ]; then head="$(git_in "$prefix/$path" rev-parse HEAD || true)"; fi
    if [ "$head" != "${entry##*:}" ]; then
      echo "tt-metal: submodule $path is at '$head', not the pinned ${entry##*:}" >&2
      return 1
    fi
  done
  if ! changed="$(git_in "$prefix" status --porcelain --untracked-files=no --ignore-submodules=untracked)"; then
    echo "tt-metal: git status failed in $prefix" >&2
    return 1
  fi
  if [ -n "$changed" ]; then
    echo "tt-metal: tracked files in $prefix differ from the pinned commits:" >&2
    printf '%s\n' "$changed" >&2
    return 1
  fi
  if ! grep -qxF "sfpi_version='$SFPI_VERSION'" "$prefix/tt_metal/sfpi-version" \
    || ! grep -qxF "sfpi_x86_64_debian_txz_hash='$SFPI_SHA256'" "$prefix/tt_metal/sfpi-version"; then
    echo "tt-metal: tt_metal/sfpi-version does not name sfpi $SFPI_VERSION with sha256 $SFPI_SHA256" >&2
    return 1
  fi
  echo "tt-metal: the tree and its submodules are at their pinned commits, unchanged"
}

# link_build: point the tree's build link at the build directory, as build_metal.sh does; refuse anything
# else already there.
link_build() {
  if [ -L "$prefix/build" ] && [ "$(readlink "$prefix/build")" = "$BUILD_DIR" ]; then
    return 0
  fi
  if [ -e "$prefix/build" ] || [ -L "$prefix/build" ]; then
    echo "tt-metal: $prefix/build exists and is not a link to $BUILD_DIR" >&2
    return 1
  fi
  mkdir -p "$build"
  ln -s "$BUILD_DIR" "$prefix/build"
}

# configure: run the CMake configure with the pin's flags and local_settings. With the Python bindings off,
# build_metal.sh passes no Python settings either. The configure fetches the CPM sources into the tree's cache
# and sfpi into runtime/sfpi, each checked by the hash tt-metal's CMake gives it.
configure() {
  local flags paths
  read -r -a flags <<<"$CMAKE_FLAGS"
  mapfile -t paths < <(local_settings)
  "$CMAKE" -S "$prefix" -B "$build" "${flags[@]}" "${paths[@]}"
}

# check_cache: refuse unless the configured cache holds every -D setting of CMAKE_FLAGS and of local_settings,
# and the Ninja generator, so the build is the configuration the pin records.
check_cache() {
  local cache="$build/CMakeCache.txt" flag key value actual
  local -a settings
  if [ ! -f "$cache" ]; then
    echo "tt-metal: $cache is missing" >&2
    return 1
  fi
  read -r -a settings <<<"$CMAKE_FLAGS"
  mapfile -t -O "${#settings[@]}" settings < <(local_settings)
  settings+=("-DCMAKE_GENERATOR=Ninja")
  for flag in "${settings[@]}"; do
    case "$flag" in
      -D*=*) key="${flag#-D}"; value="${key#*=}"; key="${key%%=*}" ;;
      *) continue ;;
    esac
    actual="$(grep -E "^${key}:[A-Z]+=" "$cache" | cut -d= -f2- || true)"
    if [ "$actual" != "$value" ]; then
      echo "tt-metal: $cache has $key='$actual', not the pinned '$value'" >&2
      return 1
    fi
  done
  echo "tt-metal: the CMake cache holds every pinned setting"
}

# check_sfpi_source: refuse unless tt-metal's own sfpi lookup (tt_metal/sfpi-info.sh, which its CMake runs as
# `CMAKE txz`) names the pinned txz and sha256 for this host. Without a hash for the host, tt_metal/hw/
# CMakeLists.txt:65-106 would build sfpi from source (about 10 CPU-hours by upstream's own estimate at line 85,
# outside LASSI_JOBS). The script is the
# pinned tree's own, run for its SHELL output only; its lines are read as text, never evaluated.
check_sfpi_source() {
  local out filename hash
  if ! out="$(bash "$prefix/tt_metal/sfpi-info.sh" SHELL txz 2>&1)"; then
    echo "tt-metal: tt_metal/sfpi-info.sh SHELL txz failed: $out" >&2
    return 1
  fi
  filename="$(sed -n "s/^sfpi_filename='\(.*\)'\$/\1/p" <<<"$out")"
  hash="$(sed -n "s/^sfpi_hash='\(.*\)'\$/\1/p" <<<"$out")"
  if [ "$filename" != "$SFPI_FILENAME" ] || [ "$hash" != "$SFPI_SHA256" ]; then
    echo "tt-metal: tt_metal/sfpi-info.sh names '$filename' with sha256 '$hash' for this host, not the pinned" \
      "$SFPI_FILENAME with $SFPI_SHA256; the configure would not fetch the pinned sfpi" >&2
    return 1
  fi
  echo "tt-metal: tt-metal's sfpi lookup names the pinned $SFPI_FILENAME and its sha256 for this host"
}

# check_sfpi: refuse unless the sfpi compiler the configure placed reports the pinned version, read as
# tt-metal's CMake reads it; when the configure kept its download, check that file's sha256 too. Its first
# --version line then holds ":$SFPI_VERSION" (tt_metal/hw/CMakeLists.txt:113-124), a candidate EXPECT_VERSION
# for P4.10 if the kernel compiler is the executable its toolchain checks.
check_sfpi() {
  local gpp="$prefix/runtime/sfpi/compiler/bin/riscv-tt-elf-g++" out archive=""
  if ! out="$("$gpp" --version 2>&1)"; then
    echo "tt-metal: cannot run $gpp --version" >&2
    return 1
  fi
  out="${out%%$'\n'*}"
  case "$out" in
    *":${SFPI_VERSION})"* | *":${SFPI_VERSION}["*) ;;
    *)
      echo "tt-metal: $gpp reports '$out', not sfpi $SFPI_VERSION" >&2
      return 1
      ;;
  esac
  if [ -d "$build/_deps" ]; then
    archive="$(find "$build/_deps" -type f -name "$SFPI_FILENAME" -print -quit)"
  fi
  if [ -z "$archive" ]; then
    echo "tt-metal: sfpi $SFPI_VERSION ($out); CMake checked its download against URL_HASH and kept no copy"
  elif echo "$SFPI_SHA256  $archive" | sha256sum -c --status; then
    echo "tt-metal: sfpi $SFPI_VERSION ($out); $archive has the pinned sha256"
  else
    echo "tt-metal: $archive does not have the pinned sha256 $SFPI_SHA256" >&2
    return 1
  fi
}

# record_cpm_sources: write lassi-cpm-sources.txt with one line per source directory the configure placed in
# the CPM cache: its path in the cache, then the git commit it holds and the count of tracked files that differ
# from it (CPM may patch a source), or, for a source unpacked from an archive, the sha256 of the sorted list of
# its files' sha256 lines.
record_cpm_sources() {
  local dir rel out="$prefix/lassi-cpm-sources.txt" commit changed digest
  : >"$out"
  for dir in "$cpm_cache"/*/*/; do
    if [ ! -d "$dir" ]; then continue; fi
    dir="${dir%/}"
    rel="${dir#"$cpm_cache"/}"
    if [ -e "$dir/.git" ]; then
      commit="$(git_in "$dir" rev-parse HEAD)"
      changed="$(git_in "$dir" status --porcelain --untracked-files=no | wc -l)"
      echo "$rel git $commit changed=$changed" >>"$out"
    else
      digest="$(cd "$dir" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum | sha256sum)"
      echo "$rel files-sha256 ${digest%% *}" >>"$out"
    fi
  done
  echo "tt-metal: $(wc -l <"$out") CPM sources recorded in $out:"
  cat "$out"
}

# build_tree: build the pin's TARGETS and metal_example_<name> for each pinned example, in one Ninja run, and
# nothing else: not the all or install targets, so TT-NN, the tools, and the other examples are left out.
# Ninja builds each target's dependencies (the libraries tt_metal links). Nothing built is run.
build_tree() {
  local name targets
  read -r -a targets <<<"$TARGETS"
  for name in $EXAMPLES; do
    targets+=("metal_example_$name")
  done
  "$CMAKE" --build "$build" --target "${targets[@]}" --parallel "$LASSI_JOBS"
}

# check_jit_files: refuse unless hw_toolchain placed the Wormhole linker scripts and objects the kernel JIT
# links with, under runtime/hw/ in the tree.
check_jit_files() {
  local dir pattern found
  for dir in toolchain lib; do
    pattern='*.ld'
    if [ "$dir" = lib ]; then pattern='*.o'; fi
    found="$(find "$prefix/runtime/hw/$dir/wormhole" -maxdepth 1 -type f -name "$pattern" -print -quit 2>/dev/null || true)"
    if [ -z "$found" ]; then
      echo "tt-metal: $prefix/runtime/hw/$dir/wormhole holds no $pattern file for the kernel JIT" >&2
      return 1
    fi
  done
  echo "tt-metal: the kernel JIT's Wormhole linker scripts and objects are in $prefix/runtime/hw"
}

# check_examples: refuse unless every pinned example is an executable file in the build's programming_examples
# directory. None is run.
check_examples() {
  local name missing=""
  for name in $EXAMPLES; do
    if [ ! -f "$build/programming_examples/metal_example_$name" ] \
      || [ ! -x "$build/programming_examples/metal_example_$name" ]; then
      missing+=" metal_example_$name"
    fi
  done
  if [ -n "$missing" ]; then
    echo "tt-metal: $build/programming_examples lacks$missing" >&2
    return 1
  fi
  echo "tt-metal: the pinned examples are built (none is run here): $EXAMPLES"
}

# check_linked: refuse unless the gate's example, the first of EXAMPLES, records the pinned compiler and linker
# in its .comment section (clang writes its version there and lld a "Linker:" line), so the build used the
# pinned ones. The CMake cache cannot show this: CMAKE_LINKER is a separate lookup the compiler driver does not
# use. Reads the file only; nothing is run.
check_linked() {
  local example out
  read -r example _ <<<"$EXAMPLES"
  if ! out="$(readelf -p .comment "$build/programming_examples/metal_example_$example" 2>&1)"; then
    echo "tt-metal: cannot read the .comment section of metal_example_$example: $out" >&2
    return 1
  fi
  if [[ "$out" != *"$CLANG_EXPECT"* ]] || [[ "$out" != *"Linker: "*"$LINKER_EXPECT"* ]]; then
    echo "tt-metal: metal_example_$example does not record $CLANG_EXPECT and the linker $LINKER_EXPECT:" >&2
    printf '%s\n' "$out" >&2
    return 1
  fi
  echo "tt-metal: metal_example_$example was compiled by $CLANG_EXPECT and linked by $LINKER_EXPECT"
}

# verify_install: the checks a kept install must pass: the tree at the pin, its build link, the configured
# cache, sfpi, the kernel JIT's files, the examples, and their compiler and linker.
verify_install() {
  verify_tree && [ "$(readlink "$prefix/build" || true)" = "$BUILD_DIR" ] && check_cache && check_sfpi \
    && check_jit_files && check_examples && check_linked
}

# move_aside <kind>: move whatever is at the prefix to <prefix>.<kind>-<date>, never deleting it (its removal
# is an owner decision).
move_aside() {
  local aside
  aside="$prefix.$1-$(date +%Y%m%d-%H%M%S)"
  if [ -e "$aside" ] || [ -L "$aside" ]; then
    echo "tt-metal: $aside already exists; nothing is moved" >&2
    return 1
  fi
  if ! mv -T "$prefix" "$aside"; then
    echo "tt-metal: cannot move $prefix aside to $aside" >&2
    return 1
  fi
  echo "tt-metal: moved $prefix aside to $aside; it stays until the owner decides to remove it"
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
      echo "tt-metal: $dir is not a directory, so the root-filesystem check cannot run; nothing is recorded" >&2
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
    echo "tt-metal: the root-filesystem check over $* did not run cleanly (find rc=$status); nothing is recorded" >&2
    printf '%s' "$failed" >&2
    return 1
  fi
  if [ -n "$leaked" ]; then
    echo "tt-metal: files changed in the system temporary directories during the install; nothing is recorded:" >&2
    printf '%s' "$leaked" >&2
    return 1
  fi
  echo "tt-metal: root-filesystem check passed: nothing changed under $* since the start" \
    "(find rc=$status, $denied unreadable paths skipped)"
}

# Everything this run creates outside the tree, and nothing else, is removed on exit: its start marker. An
# unfinished tree stays, with its marker file, for inspection or for a rerun at the same pin to resume.
marker=""
cleanup() {
  if [ -n "$marker" ]; then rm -f "$marker"; fi
}
trap cleanup EXIT

if [ -f "$prefix/$record_name" ] && [ "$(cat "$prefix/$record_name")" = "$(pinned_record)" ]; then
  if ! verify_install; then
    echo "tt-metal: $prefix carries the pinned record but does not check out; inspect it, move it aside" \
      "(never delete it), and rerun" >&2
    exit 1
  fi
  echo "tt-metal: $prefix already holds the pinned tt-metal $COMMIT"
  exit 0
fi

# The root-filesystem check below lists files changed after this marker.
marker="$(mktemp "$TMPDIR/tt-metal-start.XXXXXX")"
check_host
check_space "$INSTALL_KIB"

# An unfinished tree at the same pin is resumed; anything else at the prefix is moved aside, never deleted.
resume=0
if [ -e "$prefix" ] || [ -L "$prefix" ]; then
  if [ -f "$prefix/$unfinished_name" ] && [ "$(cat "$prefix/$unfinished_name")" = "$(pinned_record)" ]; then
    resume=1
    echo "tt-metal: resuming the unfinished tree at $prefix, which has the same pin"
  elif [ -f "$prefix/$record_name" ]; then
    move_aside previous
  elif [ -f "$prefix/$unfinished_name" ]; then
    move_aside unfinished
  else
    move_aside unknown
  fi
fi
if [ "$resume" = 0 ]; then
  mkdir -p "$LASSI_TOOLCHAINS"
  mkdir "$prefix"
  pinned_record >"$prefix/$unfinished_name"
fi

fetch_at "$prefix" "$URL" "$COMMIT"
fetch_submodules
verify_tree
check_sfpi_source
link_build
export CPM_SOURCE_CACHE="$cpm_cache"
configure
check_cache
check_sfpi
record_cpm_sources
build_tree
check_jit_files
check_examples
check_linked
# The build changed no tracked file of the pinned tree.
verify_tree

# Nothing may land on the root filesystem (Agent Rule 7): the install is not recorded when this user changed
# anything in the system temporary directories since the run started, or when that cannot be checked.
check_root_fs "$marker" /tmp /var/tmp

# Record: the unfinished marker, which holds the pinned record, becomes the install record.
mv -T "$prefix/$unfinished_name" "$prefix/$record_name"
echo "tt-metal: installed $PREFIX_NAME at $COMMIT in $prefix"
du -sk "$prefix"
