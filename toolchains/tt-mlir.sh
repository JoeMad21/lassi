#!/usr/bin/env bash
# Build the pinned tt-mlir tools in user space (task P5.4), as recorded in toolchains/tt-mlir.pin. On the workstation
# it runs in the LASSI WSL distro under the host profile (tools/local/host-profile.sh), which sets LASSI_ALLOWED_ROOT
# and LASSI_STOP_GIB with the gate's variables, through the long-job runner (tools/local/longjob.py); on alpha01 it
# runs through `uv run tools/rx.py job start --big`, with LASSI_ALLOWED_ROOT=/mnt/nvme10/joseph_ufl and
# LASSI_STOP_GIB=145 on its command line (P5.15). Under $LASSI_TOOLCHAINS/$PREFIX_NAME it places:
#   src/tt-mlir       tt-mlir at COMMIT with its history and the tag VERSION_TAG, which its configure describes;
#   toolchain/        TTMLIR_TOOLCHAIN_DIR: flatbuffers at FLATBUFFERS_COMMIT, installed (flatc, headers, library);
#   toolchain/venv/   TTMLIR_VENV_DIR: a venv of the pinned Python holding the tracked, hash-checked requirements;
#   src/llvm-project  LLVM at LLVM_COMMIT with tt-mlir's patch, checked by sha256 and git blob before it is applied;
#   src/flatbuffers   flatbuffers' source;
#   build/<part>      the CMake build trees of flatbuffers, llvm-project, and tt-mlir.
# It stands in for tt-mlir's environment build and env/activate. env/CMakeLists.txt would make a Python 3.12 venv
# with torch, run tt-metal's debugger installer, and fetch two sources the tools do not need
# (plans/spikes/p5-mlir-pins.md, Results 2), so the script builds the two parts the main build needs itself and sets
# the variables env/activate sets, under the prefix. tt-mlir is configured against LLVM's build tree and LLVM is
# never installed, which the space plan needs (OQ-049). tt-mlir's runtime, OpModel, bindings, tests, explorer, ttrt,
# and alchemist are off and only TARGETS (ttmlir-opt and ttmlir-translate) are built, so no tt-metal is fetched or
# built. Of what it builds, the builds run their code generators (flatc and the TableGen tools), and the script runs
# flatc with --version (check_flatc) and the two tools with --version and reads both tools' .comment sections
# (check_tools, check_linked); nothing opens a device.
# The tree is built in place: a CMake build tree holds absolute paths, so it cannot be built elsewhere and renamed.
# It carries lassi-install.unfinished until every check has passed, and only then is that file renamed to the install
# record, lassi-install.txt. Idempotent: an install whose record matches the pin and whose trees check out is kept.
# An unfinished tree of the same pin is resumed: each fetch keeps a repository already at its commit, CMake
# configures again, and Ninja continues. Anything else at the prefix (an install or unfinished tree of another
# commit, patch, flag, or tool; an install that does not check out; unknown content) is refused and left as it is,
# never moved aside or rebuilt over: a rebuild needs the stack's space again, and agents never delete on the build
# host. The first install records what the configures fetched in lassi-fetched-sources.txt and compares nothing;
# every later run refuses the install until that list is copied into toolchains/tt-mlir-fetched-sources.txt and
# committed.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=tt-mlir.pin
source "$here/tt-mlir.pin"
# No value below has a default (AGENTS.md, rules for [OPEN] choices). On alpha01 the gate sets LASSI_TOOLCHAINS,
# LASSI_SCRATCH, and LASSI_JOBS, and the rx command line gives LASSI_ALLOWED_ROOT and LASSI_STOP_GIB, the scratch
# root and the stop line in GiB (/mnt/nvme10/joseph_ufl and 145; bible, Host Facts); on the workstation the host
# profile sets all five.
set_by="by tools/local/host-profile.sh on the workstation, or on alpha01 by the gate or the rx command line"
: "${LASSI_TOOLCHAINS:?LASSI_TOOLCHAINS must be set $set_by}"
: "${LASSI_SCRATCH:?LASSI_SCRATCH must be set $set_by}"
: "${LASSI_JOBS:?LASSI_JOBS must be set $set_by}"
: "${LASSI_ALLOWED_ROOT:?LASSI_ALLOWED_ROOT must be set $set_by}"
: "${LASSI_STOP_GIB:?LASSI_STOP_GIB must be set $set_by}"
# The gate sets no HOME: it passes its own through (tools/server/gate.py, build_env), and on the build host that is
# the scratch root (plans/spikes/p0-compile-hardening.md; rx 20260923-183739-exec-7786); the host profile sets HOME
# to the scratch root. HOME is checked with the two roots, since git, CMake, and pip write under it.
: "${HOME:?HOME must be set; the gate passes its own HOME through, the scratch root on the build host}"
# The allowed root is absolute, so that it does not move with the working directory, and not /, the whole file
# system. Each root, with its links resolved, must be the allowed root or lie under it; the allowed root is quoted
# in the pattern, so a glob character in it is matched as itself (Agent Rule 7).
case "$LASSI_ALLOWED_ROOT" in
  / | [!/]*)
    echo "tt-mlir: LASSI_ALLOWED_ROOT, set $set_by, must be an absolute path other than /, not" \
      "'$LASSI_ALLOWED_ROOT'" >&2
    exit 2
    ;;
esac
for root in "$LASSI_TOOLCHAINS" "$LASSI_SCRATCH" "$HOME"; do
  case "$(realpath -m "$root")/" in
    "$LASSI_ALLOWED_ROOT"/*) ;;
    *)
      echo "tt-mlir: refusing to work outside $LASSI_ALLOWED_ROOT, the LASSI_ALLOWED_ROOT set $set_by" \
        "(Agent Rule 7): $root" >&2
      exit 2
      ;;
  esac
done
case "$LASSI_JOBS" in
  '' | *[!0-9]* | 0) echo "tt-mlir: LASSI_JOBS must be a positive integer, not '$LASSI_JOBS'" >&2; exit 2 ;;
esac
# A leading 0 is refused too, since bash arithmetic would read the value as octal.
case "$LASSI_STOP_GIB" in
  '' | *[!0-9]* | 0*)
    echo "tt-mlir: LASSI_STOP_GIB, the stop line in GiB, must be a positive integer with no leading 0, not" \
      "'$LASSI_STOP_GIB'" >&2
    exit 2
    ;;
esac
# A tool that crashes must leave no core (Agent Rule 7). The build host's core handler pipes the core to
# systemd-coredump, which ignores a core limit of 0 and stores the core on the root filesystem; the kernel aborts a
# piped core dump only when the limit is exactly 1 byte (plans/spikes/p0-sandbox-hardening.md, R6, rx
# 20260923-133451-exec-7c3d; plans/LESSONS.md). bash's `ulimit -c` counts 1024-byte blocks, so prlimit sets the limit
# in bytes, soft and hard, and the pinned Python reads it back with getrlimit; the script refuses unless it reads
# (1, 1). It comes after the root checks above, so that they refuse first, and before anything is downloaded,
# configured, or built. No step below runs a tool under a wrapper that resets the limit and re-raises a child's
# signal, as strace does (plans/LESSONS.md); every child inherits the 1-byte hard limit.
if ! prlimit --pid $$ --core=1:1 \
  || ! "$PYTHON" -I -c 'import resource, sys; sys.exit(resource.getrlimit(resource.RLIMIT_CORE) != (1, 1))'; then
  echo "tt-mlir: cannot hold RLIMIT_CORE at exactly 1 byte, soft and hard (prlimit, read back as (1, 1)); a" \
    "crashing tool could store a core on the root filesystem (Agent Rule 7)" >&2
  exit 2
fi
prefix="$LASSI_TOOLCHAINS/$PREFIX_NAME"
# Compilers, git, CMake, pip, and bash for here-documents put temporary files under TMPDIR; keep them on the
# scratch disk (Agent Rule 7).
export TMPDIR="$LASSI_SCRATCH/tmp"
mkdir -p "$TMPDIR"
# The host environment adds nothing to the pinned configuration: no compiler, flag, search-path, Python, CMake,
# CPM, or pip variable (pip's cache directory, which the gate puts on the scratch disk, aside), and no tt-metal or
# tt-mlir variable, which could name another tree.
unset CC CXX CFLAGS CXXFLAGS CPPFLAGS LDFLAGS CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH \
  LD_LIBRARY_PATH LD_RUN_PATH COMPILER_PATH CMAKE_PREFIX_PATH PKG_CONFIG_PATH PYTHONPATH PYTHONHOME \
  PYTHONUSERBASE VIRTUAL_ENV CONDA_PREFIX
for name in $(compgen -e); do
  case "$name" in
    PIP_CACHE_DIR) ;;
    TTMLIR_* | TT_MLIR_* | TT_METAL_* | CMAKE_* | CPM_* | PIP_*) unset "$name" ;;
  esac
done
# What env/activate sets (:1-4, :11-14, :24-25) and what its sourcing of the venv's activate script sets (:16-17),
# under the prefix: TTMLIR_ENV_ACTIVATED and TTMLIR_TOOLCHAIN_DIR, which tt-mlir's configure requires
# (CMakeLists.txt:12-14, :98-100); TTMLIR_VENV_DIR, whose python3 it takes (:134); and VIRTUAL_ENV, so that CMake's
# FindPython looks in the venv first (runtime/lib/ttnn/operations/CMakeLists.txt:1). The venv's bin directory comes
# first on PATH, where docs/CMakeLists.txt:17-25 finds sphinx-build and sphinx-apidoc and
# cmake/modules/TTMLIRPythonSitePackages.cmake:2 finds python; the toolchain's bin directory holds flatc.
export TTMLIR_ENV_ACTIVATED=1
export TTMLIR_TOOLCHAIN_DIR="$prefix/$TOOLCHAIN_DIR"
export TTMLIR_VENV_DIR="$prefix/$VENV_DIR"
export VIRTUAL_ENV="$TTMLIR_VENV_DIR"
export PATH="$TTMLIR_VENV_DIR/bin:$TTMLIR_TOOLCHAIN_DIR/bin:$PATH"
# pip reads no configuration file (PIP_CONFIG_FILE set to os.devnull turns them all off), so no host or user
# setting can add an index or an option.
export PIP_CONFIG_FILE=/dev/null PIP_DISABLE_PIP_VERSION_CHECK=1
record_name=lassi-install.txt
unfinished_name=lassi-install.unfinished
fetched_name=lassi-fetched-sources.txt
tracked_fetched="$here/tt-mlir-fetched-sources.txt"
ttmlir_src="$prefix/src/tt-mlir"
llvm_src="$prefix/src/llvm-project"
fb_src="$prefix/src/flatbuffers"
ttmlir_build="$prefix/build/tt-mlir"
llvm_build="$prefix/build/llvm-project"
fb_build="$prefix/build/flatbuffers"

# git_in <dir> <args>...: run git in the repository whose .git is in <dir>, never in one found above it, with
# automatic maintenance off so that no background git outlives this script.
git_in() {
  local dir="$1"
  shift
  git --git-dir="$dir/.git" --work-tree="$dir" -C "$dir" -c gc.auto=0 -c maintenance.auto=false \
    -c advice.detachedHead=false "$@"
}

# pinned_record: print the record an install from the current pin carries: every pinned commit, hash, flag, target,
# host tool, and layout value, the sha256 of the tracked requirements, and, as toolchains/tt-metal.sh records its
# own, the -D settings this script adds to each configure (flatbuffers_settings, llvm_settings, and ttmlir_settings,
# each with common_settings), among them LLD_DIR and the include flag for LLD's headers; the compilers configure()
# passes are not in those lines, but the host line names them. A later edit of the script
# that changes or drops a setting changes the record, so a kept install is then refused. It runs no build tool and
# writes nothing.
pinned_record() {
  local digest
  digest="$(sha256sum <"$here/$PYTHON_REQUIREMENTS")"
  echo "tt-mlir $COMMIT $URL $VERSION"
  echo "version-tag $VERSION_TAG $VERSION_TAG_COMMIT"
  echo "llvm $LLVM_COMMIT $LLVM_URL"
  echo "patch $LLVM_PATCH $LLVM_PATCH_BLOB $LLVM_PATCH_SHA256"
  echo "flatbuffers $FLATBUFFERS_COMMIT $FLATBUFFERS_URL"
  echo "requirements $PYTHON_REQUIREMENTS ${digest%% *}"
  echo "host $CMAKE ($CMAKE_EXPECT); $NINJA ($NINJA_EXPECT); $CLANG_C, $CLANG_CXX ($CLANG_EXPECT);" \
    "$LINKER ($LINKER_EXPECT); $PYTHON ($PYTHON_EXPECT)"
  echo "flatbuffers-flags $FLATBUFFERS_CMAKE_FLAGS"
  echo "flatbuffers-local $(flatbuffers_settings | tr '\n' ' ')"
  echo "llvm-flags $LLVM_CMAKE_FLAGS"
  echo "llvm-local $(llvm_settings | tr '\n' ' ')"
  echo "flags $CMAKE_FLAGS"
  echo "local $(ttmlir_settings | tr '\n' ' ')"
  echo "targets $TARGETS; tools $TTMLIR_OPT $TTMLIR_TRANSLATE ($EXPECT_VERSION)"
  echo "layout toolchain $TOOLCHAIN_DIR; venv $VENV_DIR; sources src/<part>; builds build/<part>"
}

# check_prefix: pass when nothing is at the prefix, or when it holds an unfinished tree of the same pin, which the
# install resumes; refuse anything else and leave it as it is. Nothing is moved aside or rebuilt over: a rebuild
# would need the stack's space again (INSTALL_KIB), and only the owner removes files on the build host.
check_prefix() {
  if [ ! -e "$prefix" ] && [ ! -L "$prefix" ]; then
    echo "tt-mlir: nothing is at $prefix yet"
    return 0
  fi
  if [ ! -L "$prefix" ] && [ -f "$prefix/$unfinished_name" ] && [ ! -e "$prefix/$record_name" ] \
    && [ "$(cat "$prefix/$unfinished_name")" = "$(pinned_record)" ]; then
    echo "tt-mlir: resuming the unfinished tree at $prefix, which has the same pin"
    return 0
  fi
  echo "tt-mlir: $prefix ($PREFIX_NAME) holds an install or unfinished tree of another pin, or unknown content;" \
    "it is refused and left as it is, since agents never delete on the build host. Inspect it and ask the owner" \
    "(plans/OWNER-QUEUE.md) before anything there is moved or removed." >&2
  return 1
}

# expect_version <expected> <command>...: refuse unless the first line the command prints holds <expected>.
expect_version() {
  local expected="$1" out
  shift
  if ! out="$("$@" 2>&1)"; then
    echo "tt-mlir: cannot run $* to read its version" >&2
    return 1
  fi
  out="${out%%$'\n'*}"
  if [[ "$out" != *"$expected"* ]]; then
    echo "tt-mlir: $1 reports '$out', not the pinned '$expected'; a pin change is a Decision Log entry" >&2
    return 1
  fi
  echo "tt-mlir: $1: $out"
}

# check_host: refuse when a host tool is not the one the pin names (Agent Rule 10): CMake and Ninja by path, the
# compilers and the linker tt-mlir takes from PATH, and the Python the venv is made from.
check_host() {
  local tool
  expect_version "$CMAKE_EXPECT" "$CMAKE" --version
  expect_version "$NINJA_EXPECT" "$NINJA" --version
  for tool in "$CLANG_C" "$CLANG_CXX" "$LINKER"; do
    if ! command -v "$tool" >/dev/null; then
      echo "tt-mlir: $tool, which the build takes from PATH, is not there" >&2
      return 1
    fi
  done
  expect_version "$CLANG_EXPECT" "$(command -v "$CLANG_C")" --version
  expect_version "$CLANG_EXPECT" "$(command -v "$CLANG_CXX")" --version
  expect_version "$LINKER_EXPECT" "$(command -v "$LINKER")" --version
  expect_version "$PYTHON_EXPECT" "$PYTHON" --version
  echo "tt-mlir: $(git --version)"
}

# check_space <planned KiB>: refuse when the planned bytes do not fit in the free space of the scratch disk, or
# when, added to what the scratch root holds now (du -s), they would pass the stop line, LASSI_STOP_GIB GiB (145 on
# both hosts, which keeps the scratch root under the owner's cap of 150 GiB; bible, Host Facts; OQ-049, 2026-10-11).
# P5.4 plans for both MLIR stacks together.
check_space() {
  local planned="$1" stop=$((LASSI_STOP_GIB * 1024 * 1024)) used free value
  used="$( (du -sk "$LASSI_SCRATCH" 2>/dev/null || true) | cut -f1)"
  free="$(df -Pk "$LASSI_SCRATCH" | awk 'NR == 2 { print $4 }')"
  for value in "$used" "$free"; do
    case "$value" in
      '' | *[!0-9]*)
        echo "tt-mlir: cannot measure the scratch disk (du -s gave '$used' KiB, df gave '$free' KiB free)" >&2
        return 1
        ;;
    esac
  done
  if [ $((used + planned)) -gt "$stop" ]; then
    echo "tt-mlir: $LASSI_SCRATCH holds $used KiB; the planned $planned KiB would pass the $LASSI_STOP_GIB GiB" \
      "stop line ($stop KiB, LASSI_STOP_GIB)" >&2
    echo "tt-mlir: queue backup or deletion candidates in the owner queue before installing" >&2
    return 1
  fi
  if [ "$planned" -gt "$free" ]; then
    echo "tt-mlir: the planned $planned KiB do not fit in the $free KiB free on the scratch disk" >&2
    return 1
  fi
  echo "tt-mlir: space check passed: $used KiB used under $LASSI_SCRATCH, $planned KiB planned, stop line $stop KiB"
}

# fetch_at <dir> <url> <commit> [<fetch option>...]: make <dir> a repository whose HEAD is <commit>, fetched from
# <url> by sha with no tags and the given options (--depth 1 for LLVM and flatbuffers; none for tt-mlir, whose
# history its configure describes). A repository already at <commit> is kept (a resumed run); one with another
# origin, or at another commit, is refused and left as it is.
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
    echo "tt-mlir: $dir fetches from another origin than $url; it is refused and left as it is" >&2
    return 1
  fi
  head="$(git_in "$dir" rev-parse -q --verify 'HEAD^{commit}' || true)"
  if [ "$head" = "$commit" ]; then
    echo "tt-mlir: $dir is already at $commit"
    return 0
  fi
  if [ -n "$head" ]; then
    echo "tt-mlir: $dir is at $head, not $commit; it is refused and left as it is" >&2
    return 1
  fi
  git_in "$dir" fetch -q --no-tags "$@" origin "$commit"
  git_in "$dir" checkout -q --detach "$commit"
  head="$(git_in "$dir" rev-parse HEAD)"
  if [ "$head" != "$commit" ]; then
    echo "tt-mlir: $dir checked out $head, not the pinned $commit" >&2
    return 1
  fi
  echo "tt-mlir: fetched $url at $commit"
}

# fetch_version_tag: give the tt-mlir clone the tag VERSION_TAG from URL, unless it already has it. With the tag
# local, cmake/modules/TTMLIRVersion.cmake:17-40 adds no remote and fetches no tags at configure.
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
    echo "tt-mlir: $dir is at '$head', not the pinned $commit" >&2
    return 1
  fi
  if ! changed="$(git_in "$dir" status --porcelain --untracked-files=no)"; then
    echo "tt-mlir: git status failed in $dir" >&2
    return 1
  fi
  if [ -n "$changed" ]; then
    echo "tt-mlir: tracked files in $dir differ from $commit:" >&2
    printf '%s\n' "$changed" >&2
    return 1
  fi
  echo "tt-mlir: $dir is at $commit, unchanged"
}

# verify_ttmlir_tree: refuse unless the tt-mlir clone is at COMMIT, unchanged, with VERSION_TAG at
# VERSION_TAG_COMMIT, and describes itself as TTMLIRVersion.cmake:44-49 asks, by that tag.
verify_ttmlir_tree() {
  local tagged described
  verify_clean "$ttmlir_src" "$COMMIT" || return 1
  tagged="$(git_in "$ttmlir_src" rev-parse -q --verify "refs/tags/$VERSION_TAG^{commit}" || true)"
  described="$(git_in "$ttmlir_src" describe --tags --match 'v[0-9]*.[0-9]*' --abbrev=0 2>/dev/null || true)"
  if [ "$tagged" != "$VERSION_TAG_COMMIT" ] || [ "$described" != "$VERSION_TAG" ]; then
    echo "tt-mlir: $ttmlir_src has $VERSION_TAG at '$tagged' and describes itself as '$described', not" \
      "$VERSION_TAG at $VERSION_TAG_COMMIT" >&2
    return 1
  fi
  echo "tt-mlir: $ttmlir_src describes itself as $VERSION_TAG ($VERSION_TAG_COMMIT)"
}

# make_venv: make the venv from the pinned Python unless it is there, and install the tracked requirements into it,
# wheels only, each download checked against its sha256 (--require-hashes).
make_venv() {
  if [ ! -x "$TTMLIR_VENV_DIR/bin/python" ]; then
    "$PYTHON" -m venv "$TTMLIR_VENV_DIR"
  fi
  "$TTMLIR_VENV_DIR/bin/python" -m pip install --require-hashes --only-binary :all: -r "$here/$PYTHON_REQUIREMENTS"
  check_venv
}

# normalized_pins: read name==version lines and print them sorted, each name lowercased with its runs of -, _, and
# . made one -, as pip compares names.
normalized_pins() {
  awk -F'==' 'NF == 2 { name = tolower($1); gsub(/[-_.]+/, "-", name); print name "==" $2 }' | LC_ALL=C sort
}

# check_venv: refuse unless the venv's Python is the pinned one and pip lists every name==version pin of the tracked
# requirements among the venv's packages.
check_venv() {
  local python="$TTMLIR_VENV_DIR/bin/python" missing
  expect_version "$PYTHON_EXPECT" "$python" --version || return 1
  missing="$(LC_ALL=C comm -23 \
    <(grep -E '^[A-Za-z0-9]' "$here/$PYTHON_REQUIREMENTS" | cut -d' ' -f1 | normalized_pins) \
    <("$python" -m pip freeze --all | normalized_pins))" || return 1
  if [ -n "$missing" ]; then
    echo "tt-mlir: the venv $TTMLIR_VENV_DIR lacks these pins of $PYTHON_REQUIREMENTS:" >&2
    printf '%s\n' "$missing" >&2
    return 1
  fi
  echo "tt-mlir: the venv holds every pin of $PYTHON_REQUIREMENTS"
}

# check_patch <file>: refuse unless the file has the pin's sha256 and git blob id for tt-mlir's LLVM patch.
check_patch() {
  local file="$1" digest blob
  if [ ! -f "$file" ]; then
    echo "tt-mlir: the LLVM patch $file is missing" >&2
    return 1
  fi
  digest="$(sha256sum <"$file")"
  digest="${digest%% *}"
  blob="$(git hash-object --no-filters -- "$file")"
  if [ "$digest" != "$LLVM_PATCH_SHA256" ] || [ "$blob" != "$LLVM_PATCH_BLOB" ]; then
    echo "tt-mlir: $file has sha256 $digest and git blob $blob, not the pinned $LLVM_PATCH_SHA256 and" \
      "$LLVM_PATCH_BLOB" >&2
    return 1
  fi
  echo "tt-mlir: $file has the pinned sha256 and git blob"
}

# patch_llvm: apply tt-mlir's patch to the LLVM tree, as env/CMakeLists.txt:55 does, once check_patch passes; a tree
# that already holds it (a resumed run) is left as it is. Upstream also commits the patch; here it stays in the work
# tree, so the tree's HEAD stays LLVM_COMMIT.
patch_llvm() {
  local patch="$ttmlir_src/$LLVM_PATCH"
  check_patch "$patch"
  if git_in "$llvm_src" apply --reverse --check "$patch" 2>/dev/null; then
    echo "tt-mlir: the LLVM tree already holds $LLVM_PATCH"
    return 0
  fi
  git_in "$llvm_src" apply "$patch"
  echo "tt-mlir: applied $LLVM_PATCH to the LLVM tree"
}

# verify_llvm_tree: refuse unless the LLVM tree is at LLVM_COMMIT with the patch applied and nothing else changed:
# the patch reverses cleanly, and the tracked files' changes count, file by file, the lines the patch adds and
# removes.
verify_llvm_tree() {
  local patch="$ttmlir_src/$LLVM_PATCH" head expected actual
  check_patch "$patch" || return 1
  head="$(git_in "$llvm_src" rev-parse HEAD 2>/dev/null || true)"
  if [ "$head" != "$LLVM_COMMIT" ]; then
    echo "tt-mlir: $llvm_src is at '$head', not the pinned $LLVM_COMMIT" >&2
    return 1
  fi
  expected="$(git_in "$llvm_src" apply --numstat "$patch" | LC_ALL=C sort)" || return 1
  actual="$(git_in "$llvm_src" diff --numstat HEAD | LC_ALL=C sort)" || return 1
  if [ "$actual" != "$expected" ] || ! git_in "$llvm_src" apply --reverse --check "$patch" 2>/dev/null; then
    echo "tt-mlir: $llvm_src is not $LLVM_COMMIT with $LLVM_PATCH alone applied; its changes:" >&2
    printf '%s\n' "$actual" >&2
    return 1
  fi
  echo "tt-mlir: $llvm_src is at $LLVM_COMMIT with $LLVM_PATCH applied and nothing else changed"
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

# llvm_settings: common_settings and the venv's Python (env/CMakeLists.txt:60). No install prefix is set: LLVM is
# never installed (its default prefix is /usr/local).
llvm_settings() {
  common_settings
  echo "-DPython3_EXECUTABLE=$TTMLIR_VENV_DIR/bin/python"
}

# ttmlir_settings: common_settings; LLVM's build tree, where cmake/modules/FindMLIR.cmake:2-3 finds MLIR and LLD
# through the config files LLVM writes for its build tree (mlir/cmake/modules/CMakeLists.txt:50-66,
# lld/cmake/modules/CMakeLists.txt:23-34, and llvm/cmake/modules/CMakeLists.txt:44-100 at LLVM_COMMIT); LLD's header
# directory, as a system directory like LLVM's (CMakeLists.txt:161), since lib/Target/LLVM/LLVMToDynamicLib.cpp:32
# includes lld/Common/Driver.h, which an LLVM install places beside LLVM's own headers (lld/CMakeLists.txt:212-218)
# but the build tree's LLVM include directories leave out (llvm/cmake/modules/CMakeLists.txt:55-61), and tt-mlir
# never adds LLD's (lld/cmake/modules/CMakeLists.txt:27-30); and flatc by path, the one
# cmake/modules/BuildFlatbuffers.cmake:1 would find on PATH.
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

# check_cache <CMakeCache.txt> <flags>: refuse unless the cache holds every -DKEY=VALUE setting of <flags>, value
# for value, and the generator a leading "-G <name>" names.
check_cache() {
  local cache="$1" word key value actual generator=0
  local -a words
  if [ ! -f "$cache" ]; then
    echo "tt-mlir: $cache is missing" >&2
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
        *) echo "tt-mlir: '$word' is not a setting check_cache reads" >&2; return 1 ;;
      esac
    fi
    actual="$(grep -E "^${key}:[A-Z]+=" "$cache" | cut -d= -f2- || true)"
    if [ "$actual" != "$value" ]; then
      echo "tt-mlir: $cache has $key='$actual', not the pinned '$value'" >&2
      return 1
    fi
  done
  echo "tt-mlir: $cache holds every pinned setting of its flags"
}

# check_compilers <CMakeCache.txt>: refuse unless the configure took the pinned compilers. CMake stores the path it
# found for a compiler given by name, so the cache is checked by name and version, not word for word.
check_compilers() {
  local cache="$1" key name actual
  for key in CMAKE_C_COMPILER CMAKE_CXX_COMPILER; do
    name="$CLANG_C"
    if [ "$key" = CMAKE_CXX_COMPILER ]; then name="$CLANG_CXX"; fi
    actual="$(grep -E "^${key}:[A-Z]+=" "$cache" | cut -d= -f2- || true)"
    if [ "${actual##*/}" != "$name" ]; then
      echo "tt-mlir: $cache has $key='$actual', not the pinned $name" >&2
      return 1
    fi
    expect_version "$CLANG_EXPECT" "$actual" --version || return 1
  done
}

# check_flatbuffers_cache, check_llvm_cache, check_ttmlir_cache: refuse unless the part's configured cache holds the
# pin's flags, the settings this script adds, and the pinned compilers.
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

# build_flatbuffers: configure flatbuffers, check its cache, and build and install it into TTMLIR_TOOLCHAIN_DIR, as
# env/CMakeLists.txt:33-46 does.
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
    echo "tt-mlir: $flatc is not an executable file" >&2
    return 1
  fi
  echo "tt-mlir: $flatc: $("$flatc" --version 2>&1 || true)"
}

# build_llvm: configure LLVM (its llvm/ directory, env/CMakeLists.txt:70), check its cache, and build its default
# target in its build tree. Nothing is installed.
build_llvm() {
  local -a settings
  mapfile -t settings < <(llvm_settings)
  configure "$llvm_src/llvm" "$llvm_build" "$LLVM_CMAKE_FLAGS" "${settings[@]}"
  check_llvm_cache
  "$CMAKE" --build "$llvm_build" --parallel "$LASSI_JOBS"
}

# configure_ttmlir: configure tt-mlir against LLVM's build tree and check its cache.
configure_ttmlir() {
  local -a settings
  mapfile -t settings < <(ttmlir_settings)
  configure "$ttmlir_src" "$ttmlir_build" "$CMAKE_FLAGS" "${settings[@]}"
  check_ttmlir_cache
}

# build_ttmlir: build the pin's TARGETS and nothing else, not the all or install targets.
build_ttmlir() {
  local -a targets
  read -r -a targets <<<"$TARGETS"
  "$CMAKE" --build "$ttmlir_build" --target "${targets[@]}" --parallel "$LASSI_JOBS"
}

# record_fetched_sources: write lassi-fetched-sources.txt in the prefix, one line per source directory a configure
# fetched: FetchContent's (_deps/<name>-src) and ExternalProject's (<name>-prefix/src/<name>) in the three build
# trees, and those tt-mlir's ExternalProject would place in its own source tree, under third_party/
# (third_party/CMakeLists.txt:9-13, :105, with the runtime or OpModel on). Each line is the directory's path in the
# prefix, then the git commit it holds and the count of tracked files that differ from it, or, for one unpacked
# from an archive, the sha256 of the sorted list of its files' sha256 lines. The one line none means nothing.
record_fetched_sources() {
  local out="$prefix/$fetched_name" found dir commit changed digest
  found="$(find "$prefix/build" "$ttmlir_src/third_party" -type d \
    \( -path '*/_deps/*-src' -o -path '*-prefix/src/*' -o -path "$ttmlir_src/third_party/*/src/*" \) -prune \
    \( -name '*-build' -o -name '*-stamp' -o -print \) | LC_ALL=C sort)"
  : >"$out"
  while IFS= read -r dir; do
    if [ -z "$dir" ]; then continue; fi
    if [ -e "$dir/.git" ]; then
      commit="$(git_in "$dir" rev-parse HEAD)"
      changed="$(git_in "$dir" status --porcelain --untracked-files=no | wc -l)"
      echo "${dir#"$prefix"/} git $commit changed=$changed" >>"$out"
    else
      digest="$(cd "$dir" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum | sha256sum)"
      echo "${dir#"$prefix"/} files-sha256 ${digest%% *}" >>"$out"
    fi
  done <<<"$found"
  if [ ! -s "$out" ]; then echo none >"$out"; fi
  echo "tt-mlir: the configures' fetched sources, recorded in $out:"
  cat "$out"
}

# tracked_list_is_placeholder: succeed when the tracked list's only entry is PLACEHOLDER, as until the first
# install's list is copied into it.
tracked_list_is_placeholder() {
  [ "$(sed -e 's/[[:space:]]*$//' -e '/^#/d' -e '/^$/d' "$tracked_fetched")" = PLACEHOLDER ]
}

# check_fetched_sources <tree list> <tracked list>: refuse unless the install's list equals the tracked one line
# for line, "#" lines, blank lines, and trailing spaces aside. A tracked list that is PLACEHOLDER records nothing,
# so it is refused too.
check_fetched_sources() {
  local tree="$1" tracked="$2" ours theirs
  if [ ! -f "$tree" ] || [ ! -f "$tracked" ]; then
    echo "tt-mlir: cannot compare the fetched sources: $tree or $tracked is missing" >&2
    return 1
  fi
  theirs="$(sed -e 's/[[:space:]]*$//' -e '/^#/d' -e '/^$/d' "$tracked")"
  if [ "$theirs" = PLACEHOLDER ]; then
    echo "tt-mlir: $tracked is PLACEHOLDER, no record; copy $tree into it and commit it first" >&2
    return 1
  fi
  ours="$(sed -e 's/[[:space:]]*$//' -e '/^#/d' -e '/^$/d' "$tree")"
  if [ "$ours" != "$theirs" ]; then
    echo "tt-mlir: $tree differs from $tracked; a change there is a pin change (Decision Log)" >&2
    diff <(printf '%s\n' "$theirs") <(printf '%s\n' "$ours") >&2 || true
    return 1
  fi
  echo "tt-mlir: $tree equals $tracked"
}

# check_tools <expected>: refuse unless TTMLIR_OPT and TTMLIR_TRANSLATE are executable files in the prefix whose
# --version output holds <expected> (EXPECT_VERSION) and whose .comment sections each record the pinned compiler and
# linker (check_linked).
check_tools() {
  local expected="$1" key tool out
  for key in TTMLIR_OPT TTMLIR_TRANSLATE; do
    tool="$prefix/${!key}"
    if [ ! -f "$tool" ] || [ ! -x "$tool" ]; then
      echo "tt-mlir: $tool ($key) is not an executable file" >&2
      return 1
    fi
    if ! out="$("$tool" --version 2>&1)" || [[ "$out" != *"$expected"* ]]; then
      echo "tt-mlir: $tool --version does not print '$expected':" >&2
      printf '%s\n' "$out" >&2
      return 1
    fi
    echo "tt-mlir: $tool --version prints '$expected'"
    check_linked "$tool" || return 1
  done
}

# check_linked <program>: refuse unless the program's .comment section records the pinned compiler and linker
# (clang writes its version there and lld a "Linker:" line). tt-mlir links with ld.lld-<clang major> only when it
# finds one (CMakeLists.txt:110-125), and with the compiler's default linker otherwise. Reads the file only.
check_linked() {
  local out
  if ! out="$(readelf -p .comment "$1" 2>&1)"; then
    echo "tt-mlir: cannot read the .comment section of $1: $out" >&2
    return 1
  fi
  if [[ "$out" != *"$CLANG_EXPECT"* ]] || [[ "$out" != *"Linker: "*"$LINKER_EXPECT"* ]]; then
    echo "tt-mlir: $1 does not record $CLANG_EXPECT and the linker $LINKER_EXPECT:" >&2
    printf '%s\n' "$out" >&2
    return 1
  fi
  echo "tt-mlir: $1 was compiled by $CLANG_EXPECT and linked by $LINKER_EXPECT"
}

# verify_sources: refuse unless tt-mlir (with its tag), flatbuffers, and LLVM (with the patch alone) are at their
# pinned commits with no other tracked file changed.
verify_sources() {
  verify_ttmlir_tree && verify_clean "$fb_src" "$FLATBUFFERS_COMMIT" && verify_llvm_tree
}

# verify_install: the checks a kept install must pass: the sources, the venv, flatc, the three configured caches,
# the configures' fetched sources against the tracked list (refused while that list is PLACEHOLDER), and the tools'
# version, compiler, and linker.
verify_install() {
  verify_sources && check_venv && check_flatc && check_flatbuffers_cache && check_llvm_cache && check_ttmlir_cache \
    && check_fetched_sources "$prefix/$fetched_name" "$tracked_fetched" && check_tools "$EXPECT_VERSION"
}

# check_root_fs <marker> <dir>...: fail when this user changed anything under the directories (the system
# temporary directories and /opt, where tt-mlir's environment would put its toolchain) since <marker>, without
# crossing into other file systems (Agent Rule 7). find's errors are kept and read: "Permission denied" lines are
# normal in shared directories and are counted, while a directory that is not there, any other error, or a failure
# with no message fails the check, so a find that could not look is never read as a clean one. find prints each
# hit as "changed <path>". The method of toolchains/tt-metal.sh (task P4.2).
check_root_fs() {
  local marker="$1" dir out line status=0 denied=0 leaked="" failed=""
  shift
  for dir in "$@"; do
    if [ ! -d "$dir" ]; then
      echo "tt-mlir: $dir is not a directory, so the root-filesystem check cannot run; nothing is recorded" >&2
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
    echo "tt-mlir: the root-filesystem check over $* did not run cleanly (find rc=$status); nothing is recorded" >&2
    printf '%s' "$failed" >&2
    return 1
  fi
  if [ -n "$leaked" ]; then
    echo "tt-mlir: files changed under $* during the install; nothing is recorded:" >&2
    printf '%s' "$leaked" >&2
    return 1
  fi
  echo "tt-mlir: root-filesystem check passed: nothing changed under $* since the start" \
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
    echo "tt-mlir: $prefix carries the pinned record but does not check out; it is left as it is (agents never" \
      "delete on the build host): inspect it, and ask the owner before anything there is moved or removed" >&2
    exit 1
  fi
  echo "tt-mlir: $prefix already holds the pinned tt-mlir $COMMIT"
  exit 0
fi

# The root-filesystem check below lists files changed after this marker.
marker="$(mktemp "$TMPDIR/tt-mlir-start.XXXXXX")"
# Every refusal comes before the first download: what is at the prefix, the host tools, and the space for both MLIR
# stacks together (neither stack's job starts while their sum passes the stop line; plans/p5-ir.md, P5.4).
# On a resume the plan is the full INSTALL_KIB + RESERVE_KIB again, though du already counts the unfinished tree, so
# its bytes are counted twice. That is kept on purpose as a fail-safe: on a resume it never plans less than a fresh
# start would.
# A resume near the stop line is therefore expected to be refused; queue it for the owner then.
check_prefix
check_host
check_space "$((INSTALL_KIB + RESERVE_KIB))"
if [ ! -e "$prefix" ]; then
  mkdir -p "$LASSI_TOOLCHAINS"
  mkdir "$prefix"
  pinned_record >"$prefix/$unfinished_name"
fi

# tt-mlir first, since its tree holds the LLVM patch; then the venv, which the LLVM and tt-mlir configures use.
fetch_at "$ttmlir_src" "$URL" "$COMMIT"
fetch_version_tag
verify_ttmlir_tree
make_venv
# From here on pip can download nothing: tt-mlir's configure runs `pip install sphinx sphinx-markdown-builder`
# unpinned (docs/CMakeLists.txt:8-14), which the venv already satisfies.
export PIP_NO_INDEX=1
fetch_at "$fb_src" "$FLATBUFFERS_URL" "$FLATBUFFERS_COMMIT" --depth 1
verify_clean "$fb_src" "$FLATBUFFERS_COMMIT"
build_flatbuffers
fetch_at "$llvm_src" "$LLVM_URL" "$LLVM_COMMIT" --depth 1
patch_llvm
verify_llvm_tree
build_llvm
configure_ttmlir
# The configures' fetches are recorded and, once the tracked list holds a record, compared before tt-mlir is built.
record_fetched_sources
if tracked_list_is_placeholder; then
  echo "tt-mlir: first install: $tracked_fetched is PLACEHOLDER, so nothing is compared now. Copy" \
    "$prefix/$fetched_name into it and commit it; until then every later run refuses this install."
else
  check_fetched_sources "$prefix/$fetched_name" "$tracked_fetched"
fi
build_ttmlir
check_tools "$EXPECT_VERSION"
# The builds changed no tracked file of the pinned sources.
verify_sources

# Nothing may land on the root filesystem (Agent Rule 7): the install is not recorded when this user changed
# anything in the system temporary directories or /opt since the run started, or when that cannot be checked.
check_root_fs "$marker" /tmp /var/tmp /opt

# Record: the unfinished marker, which holds the pinned record, becomes the install record.
mv -T "$prefix/$unfinished_name" "$prefix/$record_name"
echo "tt-mlir: installed $PREFIX_NAME at $COMMIT in $prefix"
du -sk "$prefix"
du -sk "$LASSI_SCRATCH"
