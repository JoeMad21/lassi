#!/usr/bin/env bash
# Build the pinned tt-mlir runtime with MLIR's Python bindings and ttrt in user space (task P17.48; the owner's answer
# to OQ-111, option (b)), as recorded in toolchains/tt-mlir-runtime.pin. On alpha01 it runs as one big job through
# `uv run tools/rx.py job start --big`, with LASSI_ALLOWED_ROOT, LASSI_STOP_GIB, and LASSI_FREE_FLOOR_GIB on its
# command line (plans/spikes/ttmlir-runtime.md gives the line); on the workstation it would run in the LASSI WSL
# distro under the host profile (tools/local/host-profile.sh) through the long-job runner (tools/local/longjob.py),
# with LASSI_FREE_FLOOR_GIB on the command line. Under $LASSI_TOOLCHAINS/$PREFIX_NAME it places:
#   src/tt-mlir            tt-mlir at COMMIT with its history and the tag VERSION_TAG, which its configure describes;
#     third_party/tt-metal tt-mlir's own tt-metal at METAL_COMMIT, which tt-mlir's ExternalProject clones, configures,
#                          builds with TT-NN and its Python module, and installs in its build_Release (METAL_DIR);
#   toolchain/             TTMLIR_TOOLCHAIN_DIR: flatbuffers at FLATBUFFERS_COMMIT, installed (flatc, headers, library);
#   toolchain/venv/        TTMLIR_VENV_DIR: a venv of the pinned Python with the tracked, hash-checked requirements and,
#                          once built, ttrt;
#   toolchain/wheelhouse/  the requirements' wheels, downloaded once and checked by sha256, for every later pip run;
#   src/llvm-project       LLVM at LLVM_COMMIT with tt-mlir's patch, checked by sha256 and git blob, then applied;
#   src/flatbuffers        flatbuffers' source;
#   build/<part>           the CMake build trees of flatbuffers, llvm-project (MLIR's Python bindings on), and tt-mlir
#                          (the runtime, its Python module, and the Python packages of tt-mlir and ttrt).
# It stands in for tt-mlir's environment build and env/activate, as toolchains/tt-mlir.sh does, and configures tt-mlir
# against LLVM's build tree; LLVM is never installed. The compiler install at $LASSI_TOOLCHAINS/$COMPILER_PREFIX_NAME
# is never written to or read as an input: the script refuses a layout in which either prefix holds the other, fetches
# its own sources by the same commits, and refuses its install record when anything there changed during the run.
# Every build is a Ninja run at LASSI_JOBS, at the niceness the gate or the runner gives (NICE_MIN or more), under the
# 1-byte core limit, and stops when the scratch disk's free space falls under LASSI_FREE_FLOOR_GIB. Of what it builds,
# the builds run their code generators, tt-metal's configure, and pip; the script runs flatc, the two tools, and
# sfpi's compiler with --version, imports tt-mlir's Python package, reads ttrt's installed version, and lists the
# runtime's libraries with ldd; nothing opens a device or runs a kernel.
# The tree is built in place and carries lassi-install.unfinished until every check has passed; only then is that file
# renamed to the install record, lassi-install.txt. Idempotent: an install whose record matches the pin and whose
# trees check out is kept. An unfinished tree of the same pin is resumed (each fetch keeps a repository already at its
# commit, ExternalProject keeps its finished steps, and Ninja continues). Anything else at the prefix is refused and
# left as it is, never moved aside or rebuilt over: agents never delete on the build host. The first install records
# what the builds fetched in lassi-fetched-sources.txt and compares nothing; every later run refuses the install until
# that list is copied into toolchains/tt-mlir-runtime-fetched-sources.txt and committed.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=tt-mlir-runtime.pin
source "$here/tt-mlir-runtime.pin"
# No value below has a default (AGENTS.md, rules for [OPEN] choices). On alpha01 the gate sets LASSI_TOOLCHAINS,
# LASSI_SCRATCH, and LASSI_JOBS, and the rx command line gives LASSI_ALLOWED_ROOT and LASSI_STOP_GIB (bible, Host
# Facts); on the workstation the host profile sets those five. LASSI_FREE_FLOOR_GIB, the free space in GiB that the
# scratch disk keeps for its other users, comes from the command line on both hosts.
set_by="by tools/local/host-profile.sh on the workstation, or on alpha01 by the gate or the rx command line"
: "${LASSI_TOOLCHAINS:?LASSI_TOOLCHAINS must be set $set_by}"
: "${LASSI_SCRATCH:?LASSI_SCRATCH must be set $set_by}"
: "${LASSI_JOBS:?LASSI_JOBS must be set $set_by}"
: "${LASSI_ALLOWED_ROOT:?LASSI_ALLOWED_ROOT must be set $set_by}"
: "${LASSI_STOP_GIB:?LASSI_STOP_GIB must be set $set_by}"
floor_by="on the command line, on alpha01 and on the workstation alike; it has no default"
: "${LASSI_FREE_FLOOR_GIB:?LASSI_FREE_FLOOR_GIB, the free space in GiB the scratch disk keeps, must be set $floor_by}"
: "${HOME:?HOME must be set; the gate passes its own HOME through, the scratch root on the build host}"
# The allowed root is absolute and not /; each root, with its links resolved, must be the allowed root or lie under
# it, matched literally (Agent Rule 7), as toolchains/tt-mlir.sh checks them.
case "$LASSI_ALLOWED_ROOT" in
  / | [!/]*)
    echo "tt-mlir-runtime: LASSI_ALLOWED_ROOT, set $set_by, must be an absolute path other than /, not" \
      "'$LASSI_ALLOWED_ROOT'" >&2
    exit 2
    ;;
esac
for root in "$LASSI_TOOLCHAINS" "$LASSI_SCRATCH" "$HOME"; do
  case "$(realpath -m "$root")/" in
    "$LASSI_ALLOWED_ROOT"/*) ;;
    *)
      echo "tt-mlir-runtime: refusing to work outside $LASSI_ALLOWED_ROOT, the LASSI_ALLOWED_ROOT set $set_by" \
        "(Agent Rule 7): $root" >&2
      exit 2
      ;;
  esac
done
case "$LASSI_JOBS" in
  '' | *[!0-9]* | 0) echo "tt-mlir-runtime: LASSI_JOBS must be a positive integer, not '$LASSI_JOBS'" >&2; exit 2 ;;
esac
# A leading 0 is refused too, since bash arithmetic would read the value as octal.
for name in LASSI_STOP_GIB LASSI_FREE_FLOOR_GIB; do
  case "${!name}" in
    '' | *[!0-9]* | 0*)
      echo "tt-mlir-runtime: $name, in GiB, must be a positive integer with no leading 0, not '${!name}'" >&2
      exit 2
      ;;
  esac
done
# A tool that crashes must leave no core (Agent Rule 7): prlimit sets RLIMIT_CORE to exactly 1 byte, soft and hard,
# and the pinned Python reads it back, as toolchains/tt-mlir.sh does (plans/spikes/p0-sandbox-hardening.md, R6;
# plans/LESSONS.md). It comes after the checks above and before anything is downloaded, configured, or built; every
# child inherits it.
if ! prlimit --pid $$ --core=1:1 \
  || ! "$PYTHON" -I -c 'import resource, sys; sys.exit(resource.getrlimit(resource.RLIMIT_CORE) != (1, 1))'; then
  echo "tt-mlir-runtime: cannot hold RLIMIT_CORE at exactly 1 byte, soft and hard (prlimit, read back as (1, 1));" \
    "a crashing tool could store a core on the root filesystem (Agent Rule 7)" >&2
  exit 2
fi
prefix="$LASSI_TOOLCHAINS/$PREFIX_NAME"
compiler_prefix="$LASSI_TOOLCHAINS/$COMPILER_PREFIX_NAME"
# Compilers, git, CMake, pip, and bash put temporary files under TMPDIR; keep them on the scratch disk (Agent Rule 7).
export TMPDIR="$LASSI_SCRATCH/tmp"
mkdir -p "$TMPDIR"
# The host environment adds nothing to the pinned configuration: no compiler, flag, search-path, Python, CMake, CPM,
# or pip variable (pip's cache directory aside), and no tt-metal or tt-mlir variable, which could name another tree;
# CPM_SOURCE_CACHE among them, so tt-metal's CPM cache stays in its own tree (third_party/CMakeLists.txt:9-13).
unset CC CXX CFLAGS CXXFLAGS CPPFLAGS LDFLAGS CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH \
  LD_LIBRARY_PATH LD_RUN_PATH COMPILER_PATH CMAKE_PREFIX_PATH PKG_CONFIG_PATH PYTHONPATH PYTHONHOME \
  PYTHONUSERBASE VIRTUAL_ENV CONDA_PREFIX
for name in $(compgen -e); do
  case "$name" in
    PIP_CACHE_DIR) ;;
    TTMLIR_* | TT_MLIR_* | TT_METAL_* | CMAKE_* | CPM_* | PIP_*) unset "$name" ;;
  esac
done
# What env/activate sets (:1-4, :11-17, :24-25), under the prefix, as toolchains/tt-mlir.sh sets it: the venv's bin
# first on PATH, where tt-mlir's and ttrt's build commands run `python` and pip (docs/CMakeLists.txt:9;
# tools/ttrt/CMakeLists.txt:13, :27-28), and the toolchain's bin, which holds flatc.
export TTMLIR_ENV_ACTIVATED=1
export TTMLIR_TOOLCHAIN_DIR="$prefix/$TOOLCHAIN_DIR"
export TTMLIR_VENV_DIR="$prefix/$VENV_DIR"
export VIRTUAL_ENV="$TTMLIR_VENV_DIR"
export PATH="$TTMLIR_VENV_DIR/bin:$TTMLIR_TOOLCHAIN_DIR/bin:$PATH"
# pip reads no configuration file; git never prompts (a clone that needs credentials fails instead of waiting); and
# the ExternalProject's inner `cmake --build` of tt-metal runs at LASSI_JOBS (ExternalProject.cmake:1576-1589 at CMake
# v3.31.6), as every Ninja run below does.
export PIP_CONFIG_FILE=/dev/null PIP_DISABLE_PIP_VERSION_CHECK=1 GIT_TERMINAL_PROMPT=0
export CMAKE_BUILD_PARALLEL_LEVEL="$LASSI_JOBS"
record_name=lassi-install.txt
unfinished_name=lassi-install.unfinished
fetched_name=lassi-fetched-sources.txt
tracked_fetched="$here/tt-mlir-runtime-fetched-sources.txt"
ttmlir_src="$prefix/src/tt-mlir"
llvm_src="$prefix/src/llvm-project"
fb_src="$prefix/src/flatbuffers"
ttmlir_build="$prefix/build/tt-mlir"
llvm_build="$prefix/build/llvm-project"
fb_build="$prefix/build/flatbuffers"
metal_src="$prefix/$METAL_DIR"
metal_build="$prefix/$METAL_BUILD_DIR"
wheelhouse="$prefix/$WHEELHOUSE_DIR"
floor_kib=$((LASSI_FREE_FLOOR_GIB * 1024 * 1024))

# git_in <dir> <args>...: run git in the repository whose .git is in <dir>, never in one found above it, with
# automatic maintenance off so that no background git outlives this script.
git_in() {
  local dir="$1"
  shift
  git --git-dir="$dir/.git" --work-tree="$dir" -C "$dir" -c gc.auto=0 -c maintenance.auto=false \
    -c advice.detachedHead=false "$@"
}

# pinned_record: print the record an install from the current pin carries: every pinned commit, hash, flag, target,
# host tool, and layout value, the sha256 of the tracked requirements, and the -D settings this script adds to each
# configure. A later edit of the pin or of those settings changes the record, so a kept install is then refused. It
# runs no build tool and writes nothing.
pinned_record() {
  local digest
  digest="$(sha256sum <"$here/$PYTHON_REQUIREMENTS")"
  echo "tt-mlir $COMMIT $URL $VERSION $BUILD"
  echo "version-tag $VERSION_TAG $VERSION_TAG_COMMIT"
  echo "llvm $LLVM_COMMIT $LLVM_URL"
  echo "patch $LLVM_PATCH $LLVM_PATCH_BLOB $LLVM_PATCH_SHA256"
  echo "flatbuffers $FLATBUFFERS_COMMIT $FLATBUFFERS_URL"
  echo "tt-metal $METAL_COMMIT $METAL_URL $METAL_SUBMODULES"
  echo "sfpi $SFPI_VERSION $SFPI_FILENAME $SFPI_SHA256"
  echo "requirements $PYTHON_REQUIREMENTS ${digest%% *} $PYTHON_EXTRA_INDEX; $PIP_EXPECT; setuptools $SETUPTOOLS_EXPECT"
  echo "host $CMAKE ($CMAKE_EXPECT); $NINJA ($NINJA_EXPECT); $CLANG_C, $CLANG_CXX ($CLANG_EXPECT);" \
    "$LINKER ($LINKER_EXPECT); $PYTHON ($PYTHON_EXPECT); $MPI_LIBRARY"
  echo "flatbuffers-flags $FLATBUFFERS_CMAKE_FLAGS"
  echo "flatbuffers-local $(flatbuffers_settings | tr '\n' ' ')"
  echo "llvm-flags $LLVM_CMAKE_FLAGS"
  echo "llvm-local $(llvm_settings | tr '\n' ' ')"
  echo "flags $CMAKE_FLAGS"
  echo "local $(ttmlir_settings | tr '\n' ' ')"
  echo "metal-flags $METAL_CACHE_FLAGS"
  echo "targets $METAL_DOWNLOAD_TARGET $METAL_CONFIGURE_TARGET $METAL_TARGET; $TARGETS; $TTRT_TARGET"
  echo "tools $TTMLIR_OPT $TTMLIR_TRANSLATE ($EXPECT_VERSION); $RUNTIME_LIBRARY; $RUNTIME_MODULE; $TTRT"
  echo "layout toolchain $TOOLCHAIN_DIR; venv $VENV_DIR; wheelhouse $WHEELHOUSE_DIR; tt-metal $METAL_DIR" \
    "($METAL_BUILD_DIR: $METAL_LIBRARIES); python $PYTHON_PACKAGES; sources src/<part>; builds build/<part>"
}

# refuse_compiler_prefix: refuse when this install's prefix is the compiler install's prefix, lies inside it, or holds
# it, with links resolved, so that nothing this script writes can land in the compiler install
# ($COMPILER_PREFIX_NAME), which other runs use. Reads paths only.
refuse_compiler_prefix() {
  local ours theirs
  ours="$(realpath -m "$prefix")"
  theirs="$(realpath -m "$compiler_prefix")"
  case "$ours/" in
    "$theirs"/*)
      echo "tt-mlir-runtime: $prefix is or lies inside the compiler install $compiler_prefix; refusing" >&2
      return 1
      ;;
  esac
  case "$theirs/" in
    "$ours"/*)
      echo "tt-mlir-runtime: $prefix holds the compiler install $compiler_prefix; refusing" >&2
      return 1
      ;;
  esac
  echo "tt-mlir-runtime: $prefix is apart from the compiler install $compiler_prefix"
}

# check_prefix: pass when nothing is at the prefix, or when it holds an unfinished tree of the same pin, which the
# install resumes; refuse anything else and leave it as it is (only the owner removes files on the build host).
check_prefix() {
  if [ ! -e "$prefix" ] && [ ! -L "$prefix" ]; then
    echo "tt-mlir-runtime: nothing is at $prefix yet"
    return 0
  fi
  if [ ! -L "$prefix" ] && [ -f "$prefix/$unfinished_name" ] && [ ! -e "$prefix/$record_name" ] \
    && [ "$(cat "$prefix/$unfinished_name")" = "$(pinned_record)" ]; then
    echo "tt-mlir-runtime: resuming the unfinished tree at $prefix, which has the same pin"
    return 0
  fi
  echo "tt-mlir-runtime: $prefix ($PREFIX_NAME) holds an install or unfinished tree of another pin, or unknown" \
    "content; it is refused and left as it is, since agents never delete on the build host. Inspect it and ask the" \
    "owner (plans/OWNER-QUEUE.md) before anything there is moved or removed." >&2
  return 1
}

# expect_version <expected> <command>...: refuse unless the first line the command prints holds <expected>.
expect_version() {
  local expected="$1" out
  shift
  if ! out="$("$@" 2>&1)"; then
    echo "tt-mlir-runtime: cannot run $* to read its version" >&2
    return 1
  fi
  out="${out%%$'\n'*}"
  if [[ "$out" != *"$expected"* ]]; then
    echo "tt-mlir-runtime: $1 reports '$out', not the pinned '$expected'; a pin change is a Decision Log entry" >&2
    return 1
  fi
  echo "tt-mlir-runtime: $1: $out"
}

# check_nice: refuse unless the script runs at NICE_MIN or more, as the gate and the long-job runner run jobs, so the
# builds yield the shared host's CPUs to its other users.
check_nice() {
  local level
  level="$(nice)"
  if [ "$level" -lt "$NICE_MIN" ]; then
    echo "tt-mlir-runtime: running at niceness $level, under $NICE_MIN; run it through the gate (rx job start)" \
      "or tools/local/longjob.py, which run jobs at nice 10" >&2
    return 1
  fi
  echo "tt-mlir-runtime: running at niceness $level"
}

# check_host: refuse when a host tool is not the one the pin names (Agent Rule 10), a tool the checks use is missing,
# or the MPI library tt-metal's distributed build links is absent (read only).
check_host() {
  local tool
  expect_version "$CMAKE_EXPECT" "$CMAKE" --version
  expect_version "$NINJA_EXPECT" "$NINJA" --version
  for tool in "$CLANG_C" "$CLANG_CXX" "$LINKER" git readelf ldd df du find; do
    if ! command -v "$tool" >/dev/null; then
      echo "tt-mlir-runtime: $tool, which the build or its checks take from PATH, is not there" >&2
      return 1
    fi
  done
  expect_version "$CLANG_EXPECT" "$(command -v "$CLANG_C")" --version
  expect_version "$CLANG_EXPECT" "$(command -v "$CLANG_CXX")" --version
  expect_version "$LINKER_EXPECT" "$(command -v "$LINKER")" --version
  expect_version "$PYTHON_EXPECT" "$PYTHON" --version
  if [ ! -e "$MPI_LIBRARY" ]; then
    echo "tt-mlir-runtime: $MPI_LIBRARY is missing; tt-metal's distributed build would take another MPI or fail" \
      "(tt_metal/distributed/CMakeLists.txt:37-75)" >&2
    return 1
  fi
  echo "tt-mlir-runtime: $(git --version); $MPI_LIBRARY is there"
}

# free_kib: print the KiB free on the scratch disk (df's Available column).
free_kib() {
  df -Pk "$LASSI_SCRATCH" | awk 'NR == 2 { print $4 }'
}

# check_space <planned KiB>: refuse when the planned KiB, added to what the scratch root holds now (du -s), would pass
# the stop line, LASSI_STOP_GIB GiB, or when the disk's free space would fall under the floor, LASSI_FREE_FLOOR_GIB
# GiB, once the planned KiB are written. Both values come from the command line with no default.
check_space() {
  local planned="$1" stop=$((LASSI_STOP_GIB * 1024 * 1024)) used free value
  used="$( (du -sk "$LASSI_SCRATCH" 2>/dev/null || true) | cut -f1)"
  free="$(free_kib || true)"
  for value in "$used" "$free"; do
    case "$value" in
      '' | *[!0-9]*)
        echo "tt-mlir-runtime: cannot measure the scratch disk (du -s gave '$used' KiB, df gave '$free' KiB free)" >&2
        return 1
        ;;
    esac
  done
  if [ $((used + planned)) -gt "$stop" ]; then
    echo "tt-mlir-runtime: $LASSI_SCRATCH holds $used KiB; the planned $planned KiB would pass the $LASSI_STOP_GIB" \
      "GiB stop line ($stop KiB, LASSI_STOP_GIB)" >&2
    return 1
  fi
  if [ $((planned + floor_kib)) -gt "$free" ]; then
    echo "tt-mlir-runtime: the scratch disk has $free KiB free; the planned $planned KiB and the floor of" \
      "$floor_kib KiB (LASSI_FREE_FLOOR_GIB=$LASSI_FREE_FLOOR_GIB) need $((planned + floor_kib)) KiB; ask the owner" \
      "to free space (plans/OWNER-QUEUE.md)" >&2
    return 1
  fi
  echo "tt-mlir-runtime: space check passed: $used KiB used under $LASSI_SCRATCH, $planned KiB planned, stop line" \
    "$stop KiB; $free KiB free, floor $floor_kib KiB"
}

# run_guarded <command>...: run a build command in the background and read the scratch disk's free space every
# GUARD_SECONDS while it runs. When the free space falls under the floor, stop the command with SIGTERM (Ninja passes
# it to the jobs it runs, each in its own process group) and fail, so a build never fills the shared disk; the
# unfinished tree stays for a rerun to resume once the owner has freed space. The command stays in this script's
# process group, so the gate's stop of the job still reaches it. Returns the command's status otherwise.
run_guarded() {
  local pid free second status=0
  "$@" &
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    free="$(free_kib || true)"
    case "$free" in '' | *[!0-9]*) free=0 ;; esac
    if [ "$free" -lt "$floor_kib" ]; then
      echo "tt-mlir-runtime: the scratch disk has $free KiB free, under the floor of $floor_kib KiB" \
        "(LASSI_FREE_FLOOR_GIB); stopping $*; the unfinished tree stays for a rerun to resume" >&2
      kill -TERM "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
      return 1
    fi
    # Wait up to GUARD_SECONDS, a second at a time, so a finished build is not kept waiting.
    for ((second = 0; second < GUARD_SECONDS; second++)); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
  done
  wait "$pid" || status=$?
  return "$status"
}

# fetch_at <dir> <url> <commit> [<fetch option>...]: make <dir> a repository whose HEAD is <commit>, fetched from
# <url> by sha with no tags and the given options. A repository already at <commit> is kept (a resumed run); one with
# another origin, or at another commit, is refused and left as it is.
fetch_at() {
  local dir="$1" url="$2" commit="$3" origin head
  shift 3
  mkdir -p "$dir"
  if [ ! -e "$dir/.git" ]; then
    git -c init.defaultBranch=main -C "$dir" init -q
  fi
  origin="$(git_in "$dir" remote get-url origin 2>/dev/null || true)"
  if [ -z "$origin" ]; then
    git_in "$dir" remote add origin "$url"
  elif [ "$origin" != "$url" ]; then
    echo "tt-mlir-runtime: $dir fetches from another origin than $url; it is refused and left as it is" >&2
    return 1
  fi
  head="$(git_in "$dir" rev-parse -q --verify 'HEAD^{commit}' || true)"
  if [ "$head" = "$commit" ]; then
    echo "tt-mlir-runtime: $dir is already at $commit"
    return 0
  fi
  if [ -n "$head" ]; then
    echo "tt-mlir-runtime: $dir is at $head, not $commit; it is refused and left as it is" >&2
    return 1
  fi
  git_in "$dir" fetch -q --no-tags "$@" origin "$commit"
  git_in "$dir" checkout -q --detach "$commit"
  head="$(git_in "$dir" rev-parse HEAD)"
  if [ "$head" != "$commit" ]; then
    echo "tt-mlir-runtime: $dir checked out $head, not the pinned $commit" >&2
    return 1
  fi
  echo "tt-mlir-runtime: fetched $url at $commit"
}

# fetch_version_tag: give the tt-mlir clone the tag VERSION_TAG, unless it has it, so TTMLIRVersion.cmake:17-40
# fetches no tags at configure.
fetch_version_tag() {
  if git_in "$ttmlir_src" rev-parse -q --verify "refs/tags/$VERSION_TAG" >/dev/null; then
    return 0
  fi
  git_in "$ttmlir_src" fetch -q --no-tags origin "refs/tags/$VERSION_TAG:refs/tags/$VERSION_TAG"
}

# verify_clean <dir> <commit>: refuse unless the repository in <dir> is at <commit> with no tracked file changed.
verify_clean() {
  local dir="$1" commit="$2" head changed
  head="$(git_in "$dir" rev-parse HEAD 2>/dev/null || true)"
  if [ "$head" != "$commit" ]; then
    echo "tt-mlir-runtime: $dir is at '$head', not the pinned $commit" >&2
    return 1
  fi
  if ! changed="$(git_in "$dir" status --porcelain --untracked-files=no)"; then
    echo "tt-mlir-runtime: git status failed in $dir" >&2
    return 1
  fi
  if [ -n "$changed" ]; then
    echo "tt-mlir-runtime: tracked files in $dir differ from $commit:" >&2
    printf '%s\n' "$changed" >&2
    return 1
  fi
  echo "tt-mlir-runtime: $dir is at $commit, unchanged"
}

# verify_ttmlir_tree: refuse unless the tt-mlir clone is at COMMIT, unchanged, with VERSION_TAG at
# VERSION_TAG_COMMIT, and describes itself by that tag (TTMLIRVersion.cmake:44-49). tt-metal's clone inside it,
# under third_party/, is untracked there and checked by verify_metal_tree.
verify_ttmlir_tree() {
  local tagged described
  verify_clean "$ttmlir_src" "$COMMIT" || return 1
  tagged="$(git_in "$ttmlir_src" rev-parse -q --verify "refs/tags/$VERSION_TAG^{commit}" || true)"
  described="$(git_in "$ttmlir_src" describe --tags --match 'v[0-9]*.[0-9]*' --abbrev=0 2>/dev/null || true)"
  if [ "$tagged" != "$VERSION_TAG_COMMIT" ] || [ "$described" != "$VERSION_TAG" ]; then
    echo "tt-mlir-runtime: $ttmlir_src has $VERSION_TAG at '$tagged' and describes itself as '$described', not" \
      "$VERSION_TAG at $VERSION_TAG_COMMIT" >&2
    return 1
  fi
  echo "tt-mlir-runtime: $ttmlir_src describes itself as $VERSION_TAG ($VERSION_TAG_COMMIT)"
}

# verify_metal_tree: refuse unless tt-mlir's clone of tt-metal and its submodules are at METAL_COMMIT and the
# METAL_SUBMODULES gitlinks, no tracked file differs from them, and tt_metal/sfpi-version names the pinned sfpi and
# its sha256, as toolchains/tt-metal.sh checks its own tree (verify_tree).
verify_metal_tree() {
  local entry path head changed links pinned
  head="$(git_in "$metal_src" rev-parse HEAD 2>/dev/null || true)"
  if [ "$head" != "$METAL_COMMIT" ]; then
    echo "tt-mlir-runtime: $metal_src is at '$head', not the pinned $METAL_COMMIT" >&2
    return 1
  fi
  links="$(git_in "$metal_src" ls-tree -r HEAD | awk '$2 == "commit" { print $4 ":" $3 }' | LC_ALL=C sort)"
  pinned="$(tr ' ' '\n' <<<"$METAL_SUBMODULES" | LC_ALL=C sort)"
  if [ "$links" != "$pinned" ]; then
    echo "tt-mlir-runtime: the gitlinks at $METAL_COMMIT are not the pinned METAL_SUBMODULES" >&2
    return 1
  fi
  for entry in $METAL_SUBMODULES; do
    path="${entry%%:*}"
    head=""
    if [ -e "$metal_src/$path/.git" ]; then head="$(git_in "$metal_src/$path" rev-parse HEAD || true)"; fi
    if [ "$head" != "${entry##*:}" ]; then
      echo "tt-mlir-runtime: tt-metal submodule $path is at '$head', not the pinned ${entry##*:}" >&2
      return 1
    fi
  done
  if ! changed="$(git_in "$metal_src" status --porcelain --untracked-files=no --ignore-submodules=untracked)" \
    || [ -n "$changed" ]; then
    echo "tt-mlir-runtime: tracked files in $metal_src differ from the pinned commits: $changed" >&2
    return 1
  fi
  if ! grep -qxF "sfpi_version='$SFPI_VERSION'" "$metal_src/tt_metal/sfpi-version" \
    || ! grep -qxF "sfpi_x86_64_debian_txz_hash='$SFPI_SHA256'" "$metal_src/tt_metal/sfpi-version"; then
    echo "tt-mlir-runtime: tt_metal/sfpi-version does not name sfpi $SFPI_VERSION with sha256 $SFPI_SHA256" >&2
    return 1
  fi
  echo "tt-mlir-runtime: tt-metal and its submodules are at their pinned commits, unchanged"
}

# make_venv: make the venv from the pinned Python unless it is there; download every tracked requirement into the
# wheelhouse, wheels only, each checked against its sha256 (--require-hashes), torch from PYTHON_EXTRA_INDEX; and
# install them from the wheelhouse alone. The seeded pip and setuptools stay.
make_venv() {
  local python="$TTMLIR_VENV_DIR/bin/python"
  if [ ! -x "$python" ]; then
    "$PYTHON" -m venv "$TTMLIR_VENV_DIR"
  fi
  expect_version "$PIP_EXPECT" "$python" -m pip --version
  mkdir -p "$wheelhouse"
  "$python" -m pip download --no-cache-dir --require-hashes --only-binary :all: --no-deps \
    --extra-index-url "$PYTHON_EXTRA_INDEX" -d "$wheelhouse" -r "$here/$PYTHON_REQUIREMENTS"
  "$python" -m pip install --no-index --find-links "$wheelhouse" --require-hashes --only-binary :all: --no-deps \
    -r "$here/$PYTHON_REQUIREMENTS"
  check_venv
}

# normalized_pins: read name==version lines and print them sorted, each name lowercased with its runs of -, _, and
# . made one -, as pip compares names.
normalized_pins() {
  awk -F'==' 'NF == 2 { name = tolower($1); gsub(/[-_.]+/, "-", name); print name "==" $2 }' | LC_ALL=C sort
}

# check_venv: refuse unless the venv's Python and pip are the pinned ones, setuptools is the seeded version, and pip
# lists every name==version pin of the tracked requirements among the venv's packages.
check_venv() {
  local python="$TTMLIR_VENV_DIR/bin/python" missing
  expect_version "$PYTHON_EXPECT" "$python" --version || return 1
  expect_version "$PIP_EXPECT" "$python" -m pip --version || return 1
  missing="$(LC_ALL=C comm -23 \
    <( (grep -E '^[A-Za-z0-9]' "$here/$PYTHON_REQUIREMENTS" | cut -d' ' -f1; echo "setuptools==$SETUPTOOLS_EXPECT") \
      | normalized_pins) \
    <("$python" -m pip freeze --all | normalized_pins))" || return 1
  if [ -n "$missing" ]; then
    echo "tt-mlir-runtime: the venv $TTMLIR_VENV_DIR lacks these pins:" >&2
    printf '%s\n' "$missing" >&2
    return 1
  fi
  echo "tt-mlir-runtime: the venv holds every pin of $PYTHON_REQUIREMENTS and setuptools $SETUPTOOLS_EXPECT"
}

# check_patch <file>: refuse unless the file has the pin's sha256 and git blob id for tt-mlir's LLVM patch.
check_patch() {
  local file="$1" digest blob
  if [ ! -f "$file" ]; then
    echo "tt-mlir-runtime: the LLVM patch $file is missing" >&2
    return 1
  fi
  digest="$(sha256sum <"$file")"
  digest="${digest%% *}"
  blob="$(git hash-object --no-filters -- "$file")"
  if [ "$digest" != "$LLVM_PATCH_SHA256" ] || [ "$blob" != "$LLVM_PATCH_BLOB" ]; then
    echo "tt-mlir-runtime: $file has sha256 $digest and git blob $blob, not the pinned $LLVM_PATCH_SHA256 and" \
      "$LLVM_PATCH_BLOB" >&2
    return 1
  fi
  echo "tt-mlir-runtime: $file has the pinned sha256 and git blob"
}

# patch_llvm: apply tt-mlir's patch to the LLVM tree (env/CMakeLists.txt:55) once check_patch passes; a tree that
# already holds it (a resumed run) is left as it is. The patch stays in the work tree, so HEAD stays LLVM_COMMIT.
patch_llvm() {
  local patch="$ttmlir_src/$LLVM_PATCH"
  check_patch "$patch"
  if git_in "$llvm_src" apply --reverse --check "$patch" 2>/dev/null; then
    echo "tt-mlir-runtime: the LLVM tree already holds $LLVM_PATCH"
    return 0
  fi
  git_in "$llvm_src" apply "$patch"
  echo "tt-mlir-runtime: applied $LLVM_PATCH to the LLVM tree"
}

# verify_llvm_tree: refuse unless the LLVM tree is at LLVM_COMMIT with the patch applied and nothing else changed.
verify_llvm_tree() {
  local patch="$ttmlir_src/$LLVM_PATCH" head expected actual
  check_patch "$patch" || return 1
  head="$(git_in "$llvm_src" rev-parse HEAD 2>/dev/null || true)"
  if [ "$head" != "$LLVM_COMMIT" ]; then
    echo "tt-mlir-runtime: $llvm_src is at '$head', not the pinned $LLVM_COMMIT" >&2
    return 1
  fi
  expected="$(git_in "$llvm_src" apply --numstat "$patch" | LC_ALL=C sort)" || return 1
  actual="$(git_in "$llvm_src" diff --numstat HEAD | LC_ALL=C sort)" || return 1
  if [ "$actual" != "$expected" ] || ! git_in "$llvm_src" apply --reverse --check "$patch" 2>/dev/null; then
    echo "tt-mlir-runtime: $llvm_src is not $LLVM_COMMIT with $LLVM_PATCH alone applied; its changes:" >&2
    printf '%s\n' "$actual" >&2
    return 1
  fi
  echo "tt-mlir-runtime: $llvm_src is at $LLVM_COMMIT with $LLVM_PATCH applied and nothing else changed"
}

# common_settings: the -D settings the script adds to every configure: Ninja by path, and no reading of the user or
# system CMake package registry, which could hand a configure another project's packages.
common_settings() {
  echo "-DCMAKE_MAKE_PROGRAM=$NINJA"
  echo "-DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF"
  echo "-DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF"
}

# flatbuffers_settings: common_settings and the install into TTMLIR_TOOLCHAIN_DIR (env/CMakeLists.txt:38).
flatbuffers_settings() {
  common_settings
  echo "-DCMAKE_INSTALL_PREFIX=$TTMLIR_TOOLCHAIN_DIR"
}

# llvm_settings: common_settings and the venv's Python (env/CMakeLists.txt:60), in which MLIR's binding configure
# finds pybind11 and nanobind (MLIRDetectPythonEnv.cmake:69-124 at LLVM_COMMIT). LLVM is never installed.
llvm_settings() {
  common_settings
  echo "-DPython3_EXECUTABLE=$TTMLIR_VENV_DIR/bin/python"
}

# ttmlir_settings: as toolchains/tt-mlir.sh adds them: LLVM's build tree (FindMLIR.cmake:2-3), LLD's header
# directory as a system directory (lib/Target/LLVM/LLVMToDynamicLib.cpp:32), and flatc by path.
ttmlir_settings() {
  common_settings
  echo "-DLLVM_DIR=$llvm_build/lib/cmake/llvm"
  echo "-DMLIR_DIR=$llvm_build/lib/cmake/mlir"
  echo "-DLLD_DIR=$llvm_build/lib/cmake/lld"
  echo "-DCMAKE_CXX_FLAGS=-isystem$llvm_src/lld/include"
  echo "-DFLATBUFFERS_COMPILER=$TTMLIR_TOOLCHAIN_DIR/bin/flatc"
}

# configure <source dir> <build dir> <pinned flags> [<setting>...]: run one CMake configure with the pin's flags, the
# pinned compilers by name (tt-mlir defaults to an unversioned clang, CMakeLists.txt:3-8), and the given settings.
configure() {
  local src="$1" dir="$2"
  local -a flags
  read -r -a flags <<<"$3"
  shift 3
  "$CMAKE" -S "$src" -B "$dir" "${flags[@]}" -DCMAKE_C_COMPILER="$CLANG_C" -DCMAKE_CXX_COMPILER="$CLANG_CXX" "$@"
}

# check_cache <CMakeCache.txt> <flags>: refuse unless the cache holds every -DKEY=VALUE setting of <flags>, value for
# value, and the generator a leading "-G <name>" names.
check_cache() {
  local cache="$1" word key value actual generator=0
  local -a words
  if [ ! -f "$cache" ]; then
    echo "tt-mlir-runtime: $cache is missing" >&2
    return 1
  fi
  read -r -a words <<<"$2"
  for word in "${words[@]}"; do
    if [ "$generator" = 1 ]; then
      key=CMAKE_GENERATOR value="$word" generator=0
    else
      case "$word" in
        -G) generator=1; continue ;;
        -D*=*) key="${word#-D}"; value="${key#*=}"; key="${key%%=*}"; key="${key%%:*}" ;;
        *) echo "tt-mlir-runtime: '$word' is not a setting check_cache reads" >&2; return 1 ;;
      esac
    fi
    actual="$(grep -E "^${key}:[A-Z]+=" "$cache" | cut -d= -f2- || true)"
    if [ "$actual" != "$value" ]; then
      echo "tt-mlir-runtime: $cache has $key='$actual', not the pinned '$value'" >&2
      return 1
    fi
  done
  echo "tt-mlir-runtime: $cache holds every pinned setting of its flags"
}

# check_compilers <CMakeCache.txt>: refuse unless the configure took the pinned compilers, by name and version.
check_compilers() {
  local cache="$1" key name actual
  for key in CMAKE_C_COMPILER CMAKE_CXX_COMPILER; do
    name="$CLANG_C"
    if [ "$key" = CMAKE_CXX_COMPILER ]; then name="$CLANG_CXX"; fi
    actual="$(grep -E "^${key}:[A-Z]+=" "$cache" | cut -d= -f2- || true)"
    if [ "${actual##*/}" != "$name" ]; then
      echo "tt-mlir-runtime: $cache has $key='$actual', not the pinned $name" >&2
      return 1
    fi
    expect_version "$CLANG_EXPECT" "$actual" --version || return 1
  done
}

# check_flatbuffers_cache, check_llvm_cache, check_ttmlir_cache, check_metal_cache: refuse unless the part's
# configured cache holds the pin's flags, the settings this script adds, and the pinned compilers. tt-metal's holds
# what tt-mlir's ExternalProject passes it (third_party/CMakeLists.txt:112-126), METAL_CACHE_FLAGS.
check_flatbuffers_cache() {
  local cache="$fb_build/CMakeCache.txt"
  check_cache "$cache" "$FLATBUFFERS_CMAKE_FLAGS" && check_cache "$cache" "$(flatbuffers_settings | tr '\n' ' ')" \
    && check_compilers "$cache"
}

check_llvm_cache() {
  local cache="$llvm_build/CMakeCache.txt"
  check_cache "$cache" "$LLVM_CMAKE_FLAGS" && check_cache "$cache" "$(llvm_settings | tr '\n' ' ')" \
    && check_compilers "$cache"
}

check_ttmlir_cache() {
  local cache="$ttmlir_build/CMakeCache.txt"
  check_cache "$cache" "$CMAKE_FLAGS" && check_cache "$cache" "$(ttmlir_settings | tr '\n' ' ')" \
    && check_compilers "$cache"
}

check_metal_cache() {
  local cache="$metal_build/CMakeCache.txt"
  check_cache "$cache" "$METAL_CACHE_FLAGS" && check_compilers "$cache"
}

# build_flatbuffers: configure flatbuffers, check its cache, and build and install it into TTMLIR_TOOLCHAIN_DIR.
build_flatbuffers() {
  local -a settings
  mapfile -t settings < <(flatbuffers_settings)
  configure "$fb_src" "$fb_build" "$FLATBUFFERS_CMAKE_FLAGS" "${settings[@]}"
  check_flatbuffers_cache
  "$CMAKE" --build "$fb_build" --target install --parallel "$LASSI_JOBS"
  check_flatc
}

# check_flatc: refuse unless flatc is an executable in TTMLIR_TOOLCHAIN_DIR; print its version.
check_flatc() {
  local flatc="$TTMLIR_TOOLCHAIN_DIR/bin/flatc"
  if [ ! -f "$flatc" ] || [ ! -x "$flatc" ]; then
    echo "tt-mlir-runtime: $flatc is not an executable file" >&2
    return 1
  fi
  echo "tt-mlir-runtime: $flatc: $("$flatc" --version 2>&1 || true)"
}

# build_llvm: configure LLVM (its llvm/ directory), check its cache, and build its default target in place, guarded.
build_llvm() {
  local -a settings
  mapfile -t settings < <(llvm_settings)
  configure "$llvm_src/llvm" "$llvm_build" "$LLVM_CMAKE_FLAGS" "${settings[@]}"
  check_llvm_cache
  run_guarded "$NINJA" -C "$llvm_build" -j "$LASSI_JOBS"
}

# configure_ttmlir: configure tt-mlir against LLVM's build tree and check its cache. Nothing is fetched here: the
# ExternalProject's download is a build step.
configure_ttmlir() {
  local -a settings
  mapfile -t settings < <(ttmlir_settings)
  configure "$ttmlir_src" "$ttmlir_build" "$CMAKE_FLAGS" "${settings[@]}"
  check_ttmlir_cache
}

# build_metal: tt-mlir's tt-metal, in three guarded Ninja runs, each checked before the next: the clone
# (verify_metal_tree), tt-metal's configure (its cache and sfpi), then its build, install, and build link.
build_metal() {
  run_guarded "$NINJA" -C "$ttmlir_build" -j "$LASSI_JOBS" "$METAL_DOWNLOAD_TARGET"
  verify_metal_tree
  run_guarded "$NINJA" -C "$ttmlir_build" -j "$LASSI_JOBS" "$METAL_CONFIGURE_TARGET"
  check_metal_cache
  check_sfpi
  record_and_compare_sources
  run_guarded "$NINJA" -C "$ttmlir_build" -j "$LASSI_JOBS" "$METAL_TARGET"
  check_jit_files
}

# build_ttmlir: build the pin's TARGETS, guarded, then TTRT_TARGET, whose command copies their outputs into ttrt's
# wheel and installs it in the venv; not the all or install targets.
build_ttmlir() {
  local -a targets
  read -r -a targets <<<"$TARGETS"
  run_guarded "$NINJA" -C "$ttmlir_build" -j "$LASSI_JOBS" "${targets[@]}"
  run_guarded "$NINJA" -C "$ttmlir_build" -j "$LASSI_JOBS" "$TTRT_TARGET"
}

# check_sfpi: refuse unless the sfpi compiler tt-metal's configure placed reports the pinned version (tt-metal's
# tt_metal/hw/CMakeLists.txt:108-124 reads it the same way); when the configure kept its download, check that file's
# sha256 too. As toolchains/tt-metal.sh checks its own.
check_sfpi() {
  local gpp="$metal_src/runtime/sfpi/compiler/bin/riscv-tt-elf-g++" out archive=""
  if ! out="$("$gpp" --version 2>&1)"; then
    echo "tt-mlir-runtime: cannot run $gpp --version" >&2
    return 1
  fi
  out="${out%%$'\n'*}"
  case "$out" in
    *":${SFPI_VERSION})"* | *":${SFPI_VERSION}["*) ;;
    *) echo "tt-mlir-runtime: $gpp reports '$out', not sfpi $SFPI_VERSION" >&2; return 1 ;;
  esac
  if [ -d "$metal_build/_deps" ]; then
    archive="$(find "$metal_build/_deps" -type f -name "$SFPI_FILENAME" -print -quit)"
  fi
  if [ -n "$archive" ] && ! echo "$SFPI_SHA256  $archive" | sha256sum -c --status; then
    echo "tt-mlir-runtime: $archive does not have the pinned sha256 $SFPI_SHA256" >&2
    return 1
  fi
  echo "tt-mlir-runtime: sfpi $SFPI_VERSION ($out)"
}

# check_jit_files: refuse unless tt-metal's hw_toolchain placed the Wormhole linker scripts and objects the kernel
# JIT links with, under runtime/hw/ in its tree, as toolchains/tt-metal.sh checks its own.
check_jit_files() {
  local dir pattern found
  for dir in toolchain lib; do
    pattern='*.ld'
    if [ "$dir" = lib ]; then pattern='*.o'; fi
    found="$(find "$metal_src/runtime/hw/$dir/wormhole" -maxdepth 1 -type f -name "$pattern" -print -quit \
      2>/dev/null || true)"
    if [ -z "$found" ]; then
      echo "tt-mlir-runtime: $metal_src/runtime/hw/$dir/wormhole holds no $pattern file for the kernel JIT" >&2
      return 1
    fi
  done
  echo "tt-mlir-runtime: the kernel JIT's Wormhole linker scripts and objects are in $metal_src/runtime/hw"
}

# source_line <dir>: print the fetched-sources line of a source directory: its path in the prefix, then the git
# commit it holds and the count of tracked files that differ from it, or, for one unpacked from an archive, the
# sha256 of the sorted list of its files' sha256 lines (toolchains/tt-metal-cpm-sources.txt's two forms).
source_line() {
  local dir="$1" commit changed digest
  if [ -e "$dir/.git" ]; then
    commit="$(git_in "$dir" rev-parse HEAD)"
    changed="$(git_in "$dir" status --porcelain --untracked-files=no | wc -l)"
    echo "${dir#"$prefix"/} git $commit changed=$changed"
  else
    digest="$(cd "$dir" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum | sha256sum)"
    echo "${dir#"$prefix"/} files-sha256 ${digest%% *}"
  fi
}

# record_fetched_sources: write lassi-fetched-sources.txt in the prefix: one line per source directory the builds
# fetched, FetchContent's and ExternalProject's in the three build trees and tt-mlir's third_party/ (tt-metal's
# clone), then each package tt-metal's configure placed in its CPM cache. The one line none means nothing.
record_fetched_sources() {
  local out="$prefix/$fetched_name" found dir
  found="$(find "$prefix/build" "$ttmlir_src/third_party" -type d \
    \( -path '*/_deps/*-src' -o -path '*-prefix/src/*' -o -path "$ttmlir_src/third_party/*/src/*" \) -prune \
    \( -name '*-build' -o -name '*-stamp' -o -print \) | LC_ALL=C sort)"
  : >"$out"
  while IFS= read -r dir; do
    if [ -n "$dir" ]; then source_line "$dir" >>"$out"; fi
  done <<<"$found"
  for dir in "$metal_src/.cpmcache"/*/*/; do
    if [ -d "$dir" ]; then source_line "${dir%/}" >>"$out"; fi
  done
  if [ ! -s "$out" ]; then echo none >"$out"; fi
  echo "tt-mlir-runtime: the builds' fetched sources, recorded in $out:"
  cat "$out"
}

# tracked_list_is_placeholder: succeed when the tracked list's only entry is PLACEHOLDER.
tracked_list_is_placeholder() {
  [ "$(sed -e 's/[[:space:]]*$//' -e '/^#/d' -e '/^$/d' "$tracked_fetched")" = PLACEHOLDER ]
}

# check_fetched_sources <tree list> <tracked list>: refuse unless the install's list equals the tracked one line for
# line, "#" lines, blank lines, and trailing spaces aside; a tracked list that is PLACEHOLDER is refused too.
check_fetched_sources() {
  local tree="$1" tracked="$2" ours theirs
  if [ ! -f "$tree" ] || [ ! -f "$tracked" ]; then
    echo "tt-mlir-runtime: cannot compare the fetched sources: $tree or $tracked is missing" >&2
    return 1
  fi
  theirs="$(sed -e 's/[[:space:]]*$//' -e '/^#/d' -e '/^$/d' "$tracked")"
  if [ "$theirs" = PLACEHOLDER ]; then
    echo "tt-mlir-runtime: $tracked is PLACEHOLDER, no record; copy $tree into it and commit it first" >&2
    return 1
  fi
  ours="$(sed -e 's/[[:space:]]*$//' -e '/^#/d' -e '/^$/d' "$tree")"
  if [ "$ours" != "$theirs" ]; then
    echo "tt-mlir-runtime: $tree differs from $tracked; a change there is a pin change (Decision Log)" >&2
    diff <(printf '%s\n' "$theirs") <(printf '%s\n' "$ours") >&2 || true
    return 1
  fi
  echo "tt-mlir-runtime: $tree equals $tracked"
}

# record_and_compare_sources: record the fetched sources once tt-metal's configure has fetched its CPM packages and,
# once the tracked list holds a record, compare them before tt-metal is built.
record_and_compare_sources() {
  record_fetched_sources
  if tracked_list_is_placeholder; then
    echo "tt-mlir-runtime: first install: $tracked_fetched is PLACEHOLDER, so nothing is compared now. Copy" \
      "$prefix/$fetched_name into it and commit it; until then every later run refuses this install."
  else
    check_fetched_sources "$prefix/$fetched_name" "$tracked_fetched"
  fi
}

# check_tools <expected>: refuse unless TTMLIR_OPT and TTMLIR_TRANSLATE are executable files whose --version output
# holds <expected> and whose .comment sections record the pinned compiler and linker.
check_tools() {
  local expected="$1" key tool out
  for key in TTMLIR_OPT TTMLIR_TRANSLATE; do
    tool="$prefix/${!key}"
    if [ ! -f "$tool" ] || [ ! -x "$tool" ]; then
      echo "tt-mlir-runtime: $tool ($key) is not an executable file" >&2
      return 1
    fi
    if ! out="$("$tool" --version 2>&1)" || [[ "$out" != *"$expected"* ]]; then
      echo "tt-mlir-runtime: $tool --version does not print '$expected':" >&2
      printf '%s\n' "$out" >&2
      return 1
    fi
    echo "tt-mlir-runtime: $tool --version prints '$expected'"
    check_linked "$tool" || return 1
  done
}

# check_linked <file>: refuse unless the file's .comment section records the pinned compiler and linker (clang writes
# its version there and lld a "Linker:" line). Reads the file only.
check_linked() {
  local out
  if ! out="$(readelf -p .comment "$1" 2>&1)"; then
    echo "tt-mlir-runtime: cannot read the .comment section of $1: $out" >&2
    return 1
  fi
  if [[ "$out" != *"$CLANG_EXPECT"* ]] || [[ "$out" != *"Linker: "*"$LINKER_EXPECT"* ]]; then
    echo "tt-mlir-runtime: $1 does not record $CLANG_EXPECT and the linker $LINKER_EXPECT:" >&2
    printf '%s\n' "$out" >&2
    return 1
  fi
  echo "tt-mlir-runtime: $1 was compiled by $CLANG_EXPECT and linked by $LINKER_EXPECT"
}

# check_ldd <file>: refuse unless the file is a regular file and ldd finds every library it needs. ldd lists them
# through the dynamic loader without running the file's code.
check_ldd() {
  local out
  if [ ! -f "$1" ]; then
    echo "tt-mlir-runtime: $1 is missing" >&2
    return 1
  fi
  if ! out="$(ldd "$1" 2>&1)" || [[ "$out" == *"not found"* ]]; then
    echo "tt-mlir-runtime: ldd cannot resolve every library $1 needs:" >&2
    printf '%s\n' "$out" >&2
    return 1
  fi
  echo "tt-mlir-runtime: ldd resolves every library of $1"
}

# check_runtime: refuse unless the runtime library and its Python module resolve every library they need, the
# library records the pinned compiler and linker, and tt-metal's build holds each of METAL_LIBRARIES.
check_runtime() {
  local name
  check_ldd "$prefix/$RUNTIME_LIBRARY" && check_ldd "$prefix/$RUNTIME_MODULE" \
    && check_linked "$prefix/$RUNTIME_LIBRARY" || return 1
  for name in $METAL_LIBRARIES; do
    if [ ! -f "$metal_build/lib/$name" ]; then
      echo "tt-mlir-runtime: $metal_build/lib/$name is missing" >&2
      return 1
    fi
  done
  echo "tt-mlir-runtime: tt-metal's build holds $METAL_LIBRARIES"
}

# check_python: refuse unless tt-mlir's Python package imports from PYTHON_PACKAGES and makes an MLIR context, ttrt's
# installed version is the one TTMLIRVersion.cmake gives this clone (<VERSION_TAG without v>.<commits since it>), its
# entry point is in the venv, and ldd resolves the runtime module ttrt's wheel installed. ttrt itself is not imported:
# its import loads the runtime libraries, which the post-build check runs instead.
check_python() {
  local python="$TTMLIR_VENV_DIR/bin/python" out count expected site
  if ! out="$("$python" -I -c 'import sys; sys.path.insert(0, sys.argv[1]); import ttmlir.ir as ir
ir.Context(); print("ttmlir.ir from", ir.__file__)' "$prefix/$PYTHON_PACKAGES" 2>&1)"; then
    echo "tt-mlir-runtime: tt-mlir's Python package does not import from $prefix/$PYTHON_PACKAGES:" >&2
    printf '%s\n' "$out" >&2
    return 1
  fi
  echo "tt-mlir-runtime: $out"
  count="$(git_in "$ttmlir_src" rev-list --count "$VERSION_TAG..HEAD")"
  expected="ttrt ${VERSION_TAG#v}.$count"
  out="$("$python" -I -c 'from importlib.metadata import version; print("ttrt", version("ttrt"))' 2>&1 || true)"
  if [ "$out" != "$expected" ] || [ ! -x "$prefix/$TTRT" ]; then
    echo "tt-mlir-runtime: the venv's ttrt reads '$out', not '$expected', or $prefix/$TTRT is missing" >&2
    return 1
  fi
  site="$("$python" -I -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  check_ldd "$site/ttrt/runtime/${RUNTIME_MODULE##*/}" || return 1
  echo "tt-mlir-runtime: $expected is installed in the venv"
}

# verify_sources: refuse unless tt-mlir (with its tag), flatbuffers, LLVM (with the patch alone), and tt-metal with
# its submodules are at their pinned commits with no other tracked file changed.
verify_sources() {
  verify_ttmlir_tree && verify_clean "$fb_src" "$FLATBUFFERS_COMMIT" && verify_llvm_tree && verify_metal_tree
}

# verify_install: the checks a kept install must pass: the sources, the venv, flatc, the four configured caches, the
# fetched sources against the tracked list (refused while that list is PLACEHOLDER), the tools, the runtime, and the
# Python packages.
verify_install() {
  verify_sources && check_venv && check_flatc && check_flatbuffers_cache && check_llvm_cache \
    && check_ttmlir_cache && check_metal_cache && check_fetched_sources "$prefix/$fetched_name" "$tracked_fetched" \
    && check_tools "$EXPECT_VERSION" && check_runtime && check_python
}

# check_compiler_untouched <marker>: refuse when anything under the compiler install changed since <marker>, the
# run's start, without crossing into other file systems. This script writes nothing there; a change means another
# process did, and the record waits for someone to look.
check_compiler_untouched() {
  local marker="$1" changed
  if [ ! -e "$compiler_prefix" ]; then
    echo "tt-mlir-runtime: no compiler install at $compiler_prefix to compare"
    return 0
  fi
  if ! changed="$(find "$compiler_prefix" -xdev -newer "$marker" -print -quit 2>&1)"; then
    echo "tt-mlir-runtime: cannot read $compiler_prefix: $changed" >&2
    return 1
  fi
  if [ -n "$changed" ]; then
    echo "tt-mlir-runtime: $compiler_prefix changed during this run (first: $changed); nothing is recorded" >&2
    return 1
  fi
  echo "tt-mlir-runtime: nothing under the compiler install $compiler_prefix changed during this run"
}

# check_root_fs <marker> <dir>...: fail when this user changed anything under the directories (the system temporary
# directories and /opt) since <marker>, without crossing into other file systems (Agent Rule 7). find's errors are
# kept and read: "Permission denied" lines are normal in shared directories and are counted, while a directory that
# is not there, any other error, or a failure with no message fails the check. The method of toolchains/tt-metal.sh.
check_root_fs() {
  local marker="$1" dir out line status=0 denied=0 leaked="" failed=""
  shift
  for dir in "$@"; do
    if [ ! -d "$dir" ]; then
      echo "tt-mlir-runtime: $dir is not a directory, so the root-filesystem check cannot run; nothing is recorded" >&2
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
    echo "tt-mlir-runtime: the root-filesystem check over $* did not run cleanly (find rc=$status); nothing is" \
      "recorded" >&2
    printf '%s' "$failed" >&2
    return 1
  fi
  if [ -n "$leaked" ]; then
    echo "tt-mlir-runtime: files changed under $* during the install; nothing is recorded:" >&2
    printf '%s' "$leaked" >&2
    return 1
  fi
  echo "tt-mlir-runtime: root-filesystem check passed: nothing changed under $* since the start" \
    "(find rc=$status, $denied unreadable paths skipped)"
}

# Everything this run creates outside the tree, and nothing else, is removed on exit: its start marker. An unfinished
# tree stays, with its marker file, for inspection or for a rerun at the same pin to resume.
marker=""
cleanup() {
  if [ -n "$marker" ]; then rm -f "$marker"; fi
}
trap cleanup EXIT

refuse_compiler_prefix
if [ -f "$prefix/$record_name" ] && [ "$(cat "$prefix/$record_name")" = "$(pinned_record)" ]; then
  if ! verify_install; then
    echo "tt-mlir-runtime: $prefix carries the pinned record but does not check out; it is left as it is (agents" \
      "never delete on the build host): inspect it, and ask the owner before anything there is moved or removed" >&2
    exit 1
  fi
  echo "tt-mlir-runtime: $prefix already holds the pinned tt-mlir runtime $COMMIT"
  exit 0
fi

# The root-filesystem and compiler-install checks below list files changed after this marker.
marker="$(mktemp "$TMPDIR/tt-mlir-runtime-start.XXXXXX")"
# Every refusal comes before the first download: what is at the prefix, the niceness, the host tools, and the space.
# On a resume the plan is the full INSTALL_KIB again, though du already counts the unfinished tree: a fail-safe, as in
# toolchains/tt-mlir.sh, so a resume near the stop line or the floor is refused and goes to the owner.
check_prefix
check_nice
check_host
check_space "$INSTALL_KIB"
if [ ! -e "$prefix" ]; then
  mkdir -p "$LASSI_TOOLCHAINS"
  mkdir "$prefix"
  pinned_record >"$prefix/$unfinished_name"
fi

# tt-mlir first, since its tree holds the LLVM patch; then the venv, which the configures use.
fetch_at "$ttmlir_src" "$URL" "$COMMIT"
fetch_version_tag
verify_ttmlir_tree
make_venv
# From here on pip downloads nothing: tt-mlir's configure (docs/CMakeLists.txt:8-14) and ttrt's build
# (tools/ttrt/CMakeLists.txt:13, :27-28) run pip, which the venv and the wheelhouse satisfy.
export PIP_NO_INDEX=1 PIP_FIND_LINKS="$wheelhouse"
fetch_at "$fb_src" "$FLATBUFFERS_URL" "$FLATBUFFERS_COMMIT" --depth 1
verify_clean "$fb_src" "$FLATBUFFERS_COMMIT"
build_flatbuffers
fetch_at "$llvm_src" "$LLVM_URL" "$LLVM_COMMIT" --depth 1
patch_llvm
verify_llvm_tree
build_llvm
configure_ttmlir
# tt-metal's clone, configure (its CPM packages and sfpi), fetched-sources record, and build, each checked.
build_metal
build_ttmlir
check_tools "$EXPECT_VERSION"
check_runtime
check_python
# The builds changed no tracked file of the pinned sources, and nothing under the compiler install.
verify_sources
check_compiler_untouched "$marker"

# Nothing may land on the root filesystem (Agent Rule 7): the install is not recorded when this user changed anything
# in the system temporary directories or /opt since the run started, or when that cannot be checked.
check_root_fs "$marker" /tmp /var/tmp /opt

# Record: the unfinished marker, which holds the pinned record, becomes the install record.
mv -T "$prefix/$unfinished_name" "$prefix/$record_name"
echo "tt-mlir-runtime: installed $PREFIX_NAME at $COMMIT in $prefix"
du -sk "$prefix"
du -sk "$LASSI_SCRATCH"
df -Pk "$LASSI_SCRATCH"
