#!/usr/bin/env bash
# Build the pinned Polygeist (cgeist) and its own LLVM in user space (task P5.5), as recorded in
# toolchains/polygeist.pin. On the workstation it runs in the LASSI WSL distro under the host profile
# (tools/local/host-profile.sh), which sets LASSI_ALLOWED_ROOT and LASSI_STOP_GIB with the gate's variables, through
# the long-job runner (tools/local/longjob.py); on alpha01 it runs through `uv run tools/rx.py job start --big`, with
# LASSI_ALLOWED_ROOT=/mnt/nvme10/joseph_ufl and LASSI_STOP_GIB=145 on its command line (P5.15). Either way it runs as
# one big job at a time (the gate's limit, or the runner's lock), never beside tt-mlir's (task P5.4). Under
# $LASSI_TOOLCHAINS/$PREFIX_NAME it places:
#   src/polygeist     Polygeist at COMMIT, depth 1; its tree records LLVM_COMMIT as its llvm-project gitlink;
#   src/llvm-project  llvm-project at LLVM_COMMIT, depth 1, fetched by commit, never through `git submodule`;
#   build/            one in-tree CMake build of LLVM, clang, MLIR, and Polygeist (Polygeist README.md:89-95), whose
#                     bin/ holds TARGETS and whose lib/clang/18/include holds clang's builtin headers.
# The PGO data directory compiled into Polygeist's passes (upstream default /var/tmp/polygeist/pgo/) is set to pgo/ in
# the prefix; nothing creates or writes it, since P5 runs no PGO pass (polygeist.pin, LLVM_CMAKE_FLAGS).
# Only TARGETS (cgeist, polygeist-opt, mlir-opt, mlir-translate, clang, and LLVMFrontendOpenACC for P12's flang) are
# built, and nothing is installed: P5
# runs the tools from the build tree, and the TT raiser (task P5.8) builds against it later (the space plan of
# OQ-049). Of what it builds, the build runs its code generators (the TableGen tools), and the script runs the five
# tools with --version (check_tools); nothing opens a device. cgeist stops at MLIR in P5's use (-S), so it never
# writes its /tmp file (polygeist.pin, CGEIST).
# The tree is built in place: a CMake build tree holds absolute paths, so it cannot be built elsewhere and renamed.
# It carries lassi-install.unfinished until every check has passed, and only then is that file renamed to the install
# record, lassi-install.txt. Idempotent: an install whose record matches the pin and whose trees check out is kept.
# An unfinished tree of the same pin is resumed: each fetch keeps a repository already at its commit, CMake
# configures again, and Ninja continues. Anything else at the prefix (an install or unfinished tree of another
# commit, flag, target, or tool; an install that does not check out; unknown content) is refused and left as it is,
# never moved aside or rebuilt over: a rebuild needs the stack's space again, and agents never delete on the build
# host. The space check plans for both MLIR stacks together: this stack alone once tt-mlir's install is complete
# (tt_mlir_installed), since du then counts tt-mlir's, and this stack with tt-mlir's reserve otherwise. The first
# install records what the configure fetched in lassi-fetched-sources.txt and compares nothing; every later run
# refuses the install until that list is copied into toolchains/polygeist-fetched-sources.txt and committed.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=polygeist.pin
source "$here/polygeist.pin"
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
# to the scratch root. HOME is checked with the two roots, since git and CMake write under it.
: "${HOME:?HOME must be set; the gate passes its own HOME through, the scratch root on the build host}"
# The allowed root is absolute, so that it does not move with the working directory, and not /, the whole file
# system. Each root, with its links resolved, must be the allowed root or lie under it; the allowed root is quoted
# in the pattern, so a glob character in it is matched as itself (Agent Rule 7).
case "$LASSI_ALLOWED_ROOT" in
  / | [!/]*)
    echo "polygeist: LASSI_ALLOWED_ROOT, set $set_by, must be an absolute path other than /, not" \
      "'$LASSI_ALLOWED_ROOT'" >&2
    exit 2
    ;;
esac
for root in "$LASSI_TOOLCHAINS" "$LASSI_SCRATCH" "$HOME"; do
  case "$(realpath -m "$root")/" in
    "$LASSI_ALLOWED_ROOT"/*) ;;
    *)
      echo "polygeist: refusing to work outside $LASSI_ALLOWED_ROOT, the LASSI_ALLOWED_ROOT set $set_by" \
        "(Agent Rule 7): $root" >&2
      exit 2
      ;;
  esac
done
case "$LASSI_JOBS" in
  '' | *[!0-9]* | 0) echo "polygeist: LASSI_JOBS must be a positive integer, not '$LASSI_JOBS'" >&2; exit 2 ;;
esac
# A leading 0 is refused too, since bash arithmetic would read the value as octal.
case "$LASSI_STOP_GIB" in
  '' | *[!0-9]* | 0*)
    echo "polygeist: LASSI_STOP_GIB, the stop line in GiB, must be a positive integer with no leading 0, not" \
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
  echo "polygeist: cannot hold RLIMIT_CORE at exactly 1 byte, soft and hard (prlimit, read back as (1, 1)); a" \
    "crashing tool could store a core on the root filesystem (Agent Rule 7)" >&2
  exit 2
fi
prefix="$LASSI_TOOLCHAINS/$PREFIX_NAME"
# Compilers, git, CMake, and bash for here-documents put temporary files under TMPDIR; keep them on the scratch disk
# (Agent Rule 7).
export TMPDIR="$LASSI_SCRATCH/tmp"
mkdir -p "$TMPDIR"
# The host environment adds nothing to the pinned configuration: no compiler, flag, search-path, Python, or CMake
# variable, and no Polygeist variable (POLYGEIST_PGO_DATA_DIR and the like, which the passes read at run time).
unset CC CXX CFLAGS CXXFLAGS CPPFLAGS LDFLAGS CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH \
  LD_LIBRARY_PATH LD_RUN_PATH COMPILER_PATH CMAKE_PREFIX_PATH PKG_CONFIG_PATH PYTHONPATH PYTHONHOME \
  PYTHONUSERBASE VIRTUAL_ENV CONDA_PREFIX
for name in $(compgen -e); do
  case "$name" in
    CMAKE_* | POLYGEIST_*) unset "$name" ;;
  esac
done
record_name=lassi-install.txt
unfinished_name=lassi-install.unfinished
fetched_name=lassi-fetched-sources.txt
tracked_fetched="$here/polygeist-fetched-sources.txt"
polygeist_src="$prefix/src/polygeist"
llvm_src="$prefix/src/llvm-project"
llvm_build="$prefix/build"
pgo_dir="$prefix/pgo"

# git_in <dir> <args>...: run git in the repository whose .git is in <dir>, never in one found above it, with
# automatic maintenance off so that no background git outlives this script.
git_in() {
  local dir="$1"
  shift
  git --git-dir="$dir/.git" --work-tree="$dir" -C "$dir" -c gc.auto=0 -c maintenance.auto=false \
    -c advice.detachedHead=false "$@"
}

# pinned_record: print the record an install from the current pin carries: both pinned commits, the flags, targets,
# host tools, built tools and their versions, and, as toolchains/tt-mlir.sh records its own, the -D settings this
# script adds to the configure (llvm_settings), among them the Polygeist source and PGO directories; the compilers
# configure_llvm passes are not in that line, but the host line names them. A later edit of the script that changes
# or drops a setting changes the record, so a kept install is then refused. It runs no build tool and writes nothing.
pinned_record() {
  echo "polygeist $COMMIT $URL $VERSION"
  echo "llvm $LLVM_COMMIT $LLVM_URL $LLVM_SUBMODULE $LLVM_VERSION"
  echo "host $CMAKE ($CMAKE_EXPECT); $NINJA ($NINJA_EXPECT); $CLANG_C, $CLANG_CXX ($CLANG_EXPECT);" \
    "$PYTHON ($PYTHON_EXPECT)"
  echo "llvm-flags $LLVM_CMAKE_FLAGS"
  echo "llvm-local $(llvm_settings | tr '\n' ' ')"
  echo "targets $TARGETS"
  echo "tools $CGEIST $POLYGEIST_OPT $MLIR_OPT $MLIR_TRANSLATE ($EXPECT_VERSION); $LLVM_CLANG ($LLVM_CLANG_EXPECT)"
  echo "layout sources src/polygeist src/llvm-project; build build; pgo pgo"
}

# check_prefix: pass when nothing is at the prefix, or when it holds an unfinished tree of the same pin, which the
# install resumes; refuse anything else and leave it as it is. Nothing is moved aside or rebuilt over: a rebuild
# would need the stack's space again (INSTALL_KIB), and only the owner removes files on the build host.
check_prefix() {
  if [ ! -e "$prefix" ] && [ ! -L "$prefix" ]; then
    echo "polygeist: nothing is at $prefix yet"
    return 0
  fi
  if [ ! -L "$prefix" ] && [ -f "$prefix/$unfinished_name" ] && [ ! -e "$prefix/$record_name" ] \
    && [ "$(cat "$prefix/$unfinished_name")" = "$(pinned_record)" ]; then
    echo "polygeist: resuming the unfinished tree at $prefix, which has the same pin"
    return 0
  fi
  echo "polygeist: $prefix ($PREFIX_NAME) holds an install or unfinished tree of another pin, or unknown content;" \
    "it is refused and left as it is, since agents never delete on the build host. Inspect it and ask the owner" \
    "(plans/OWNER-QUEUE.md) before anything there is moved or removed." >&2
  return 1
}

# tt_mlir_installed: succeed when task P5.4's tt-mlir install is complete: its prefix holds lassi-install.txt, whose
# first line names tt-mlir and tt-mlir.pin's COMMIT, and no lassi-install.unfinished; fail otherwise, printing why.
# tt-mlir.pin is read in a subshell, so no value of polygeist.pin changes; the tt-mlir prefix is only read. It decides
# only the space plan (polygeist.pin, INSTALL_KIB and RESERVE_KIB): a missing, unfinished, or other tt-mlir tree
# holds tt-mlir's whole stack in reserve, never less.
tt_mlir_installed() {
  local pinned tt_prefix tt_commit name="" commit=""
  if ! pinned="$(source "$here/tt-mlir.pin" && printf '%s %s\n' "$PREFIX_NAME" "$COMMIT")"; then
    echo "polygeist: cannot read $here/tt-mlir.pin, so tt-mlir's install is taken as not complete"
    return 1
  fi
  tt_prefix="$LASSI_TOOLCHAINS/${pinned%% *}"
  tt_commit="${pinned#* }"
  if [ ! -f "$tt_prefix/$record_name" ] || [ -e "$tt_prefix/$unfinished_name" ]; then
    echo "polygeist: tt-mlir's install at $tt_prefix is not complete (no $record_name, or $unfinished_name is" \
      "there), so the space plan holds tt-mlir's stack in reserve (RESERVE_KIB)"
    return 1
  fi
  read -r name commit _ <"$tt_prefix/$record_name" || true
  if [ "$name" != tt-mlir ] || [ "$commit" != "$tt_commit" ]; then
    echo "polygeist: $tt_prefix/$record_name names '$name $commit', not tt-mlir at its pinned $tt_commit, so the" \
      "space plan holds tt-mlir's stack in reserve (RESERVE_KIB)"
    return 1
  fi
  echo "polygeist: tt-mlir's install at $tt_prefix is complete, at $tt_commit; du counts that stack"
}

# expect_version <expected> <command>...: refuse unless the first line the command prints holds <expected>.
expect_version() {
  local expected="$1" out
  shift
  if ! out="$("$@" 2>&1)"; then
    echo "polygeist: cannot run $* to read its version" >&2
    return 1
  fi
  out="${out%%$'\n'*}"
  if [[ "$out" != *"$expected"* ]]; then
    echo "polygeist: $1 reports '$out', not the pinned '$expected'; a pin change is a Decision Log entry" >&2
    return 1
  fi
  echo "polygeist: $1: $out"
}

# check_host: refuse when a host tool is not the one the pin names (Agent Rule 10): CMake and Ninja by path, the
# compilers the configure takes by name from PATH, and the Python LLVM's configure uses.
check_host() {
  local tool
  expect_version "$CMAKE_EXPECT" "$CMAKE" --version
  expect_version "$NINJA_EXPECT" "$NINJA" --version
  for tool in "$CLANG_C" "$CLANG_CXX"; do
    if ! command -v "$tool" >/dev/null; then
      echo "polygeist: $tool, which the configure takes from PATH, is not there" >&2
      return 1
    fi
  done
  expect_version "$CLANG_EXPECT" "$(command -v "$CLANG_C")" --version
  expect_version "$CLANG_EXPECT" "$(command -v "$CLANG_CXX")" --version
  expect_version "$PYTHON_EXPECT" "$PYTHON" --version
  echo "polygeist: $(git --version)"
}

# check_space <planned KiB>: refuse when the planned bytes do not fit in the free space of the scratch disk, or
# when, added to what the scratch root holds now (du -s), they would pass the stop line, LASSI_STOP_GIB GiB (145 on
# both hosts, which keeps the scratch root under the owner's cap of 150 GiB; bible, Host Facts; OQ-049, 2026-10-11).
# The main steps plan both MLIR stacks together (tt_mlir_installed).
check_space() {
  local planned="$1" stop=$((LASSI_STOP_GIB * 1024 * 1024)) used free value
  used="$( (du -sk "$LASSI_SCRATCH" 2>/dev/null || true) | cut -f1)"
  free="$(df -Pk "$LASSI_SCRATCH" | awk 'NR == 2 { print $4 }')"
  for value in "$planned" "$used" "$free"; do
    case "$value" in
      '' | *[!0-9]*)
        echo "polygeist: cannot plan the space (planned '$planned' KiB; du -s gave '$used' KiB, df gave" \
          "'$free' KiB free)" >&2
        return 1
        ;;
    esac
  done
  if [ $((used + planned)) -gt "$stop" ]; then
    echo "polygeist: $LASSI_SCRATCH holds $used KiB; the planned $planned KiB would pass the $LASSI_STOP_GIB GiB" \
      "stop line ($stop KiB, LASSI_STOP_GIB)" >&2
    echo "polygeist: queue backup or deletion candidates in the owner queue before installing" >&2
    return 1
  fi
  if [ "$planned" -gt "$free" ]; then
    echo "polygeist: the planned $planned KiB do not fit in the $free KiB free on the scratch disk" >&2
    return 1
  fi
  echo "polygeist: space check passed: $used KiB used under $LASSI_SCRATCH, $planned KiB planned, stop line $stop KiB"
}

# fetch_at <dir> <url> <commit> [<fetch option>...]: make <dir> a repository whose HEAD is <commit>, fetched from
# <url> by sha with no tags and the given options (--depth 1 for both sources). A repository already at <commit> is
# kept (a resumed run); one with another origin, or at another commit, is refused and left as it is.
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
    echo "polygeist: $dir fetches from another origin than $url; it is refused and left as it is" >&2
    return 1
  fi
  head="$(git_in "$dir" rev-parse -q --verify 'HEAD^{commit}' || true)"
  if [ "$head" = "$commit" ]; then
    echo "polygeist: $dir is already at $commit"
    return 0
  fi
  if [ -n "$head" ]; then
    echo "polygeist: $dir is at $head, not $commit; it is refused and left as it is" >&2
    return 1
  fi
  git_in "$dir" fetch -q --no-tags "$@" origin "$commit"
  git_in "$dir" checkout -q --detach "$commit"
  head="$(git_in "$dir" rev-parse HEAD)"
  if [ "$head" != "$commit" ]; then
    echo "polygeist: $dir checked out $head, not the pinned $commit" >&2
    return 1
  fi
  echo "polygeist: fetched $url at $commit"
}

# verify_clean <dir> <commit>: refuse unless the repository in <dir> is at <commit> with no tracked file changed.
verify_clean() {
  local dir="$1" commit="$2" head changed
  head="$(git_in "$dir" rev-parse HEAD 2>/dev/null || true)"
  if [ "$head" != "$commit" ]; then
    echo "polygeist: $dir is at '$head', not the pinned $commit" >&2
    return 1
  fi
  if ! changed="$(git_in "$dir" status --porcelain --untracked-files=no)"; then
    echo "polygeist: git status failed in $dir" >&2
    return 1
  fi
  if [ -n "$changed" ]; then
    echo "polygeist: tracked files in $dir differ from $commit:" >&2
    printf '%s\n' "$changed" >&2
    return 1
  fi
  echo "polygeist: $dir is at $commit, unchanged"
}

# check_gitlink <repository dir>: refuse unless the Polygeist tree in <dir> records LLVM_COMMIT as the gitlink at
# LLVM_SUBMODULE (.gitmodules:1-3), so the LLVM fetched beside it is the one this Polygeist commit names.
check_gitlink() {
  local dir="$1" entry
  entry="$(git_in "$dir" ls-tree HEAD -- "$LLVM_SUBMODULE" 2>/dev/null || true)"
  if [ "$entry" != "160000 commit $LLVM_COMMIT"$'\t'"$LLVM_SUBMODULE" ]; then
    echo "polygeist: $dir records '$entry' at $LLVM_SUBMODULE, not the gitlink $LLVM_COMMIT the pin names" >&2
    return 1
  fi
  echo "polygeist: $dir records $LLVM_COMMIT as its $LLVM_SUBMODULE gitlink"
}

# llvm_settings: the -D settings the script adds to the one configure: Ninja by path; no reading of the user or
# system CMake package registry, which could hand a configure another project's packages; the pinned Python
# (llvm/CMakeLists.txt:872-873); Polygeist's source directory (README.md:92); and the PGO data directory in the prefix
# instead of /var/tmp (CMakeLists.txt:44). No install prefix is set: LLVM is never installed.
llvm_settings() {
  echo "-DCMAKE_MAKE_PROGRAM=$NINJA"
  echo "-DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF"
  echo "-DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF"
  echo "-DPython3_EXECUTABLE=$PYTHON"
  echo "-DLLVM_EXTERNAL_POLYGEIST_SOURCE_DIR=$polygeist_src"
  echo "-DPOLYGEIST_PGO_DEFAULT_DATA_DIR=$pgo_dir"
}

# check_cache <CMakeCache.txt> <flags>: refuse unless the cache holds every -DKEY=VALUE setting of <flags>, value
# for value, and the generator a leading "-G <name>" names.
check_cache() {
  local cache="$1" word key value actual generator=0
  local -a words
  if [ ! -f "$cache" ]; then
    echo "polygeist: $cache is missing" >&2
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
        *) echo "polygeist: '$word' is not a setting check_cache reads" >&2; return 1 ;;
      esac
    fi
    actual="$(grep -E "^${key}:[A-Z]+=" "$cache" | cut -d= -f2- || true)"
    if [ "$actual" != "$value" ]; then
      echo "polygeist: $cache has $key='$actual', not the pinned '$value'" >&2
      return 1
    fi
  done
  echo "polygeist: $cache holds every pinned setting of its flags"
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
      echo "polygeist: $cache has $key='$actual', not the pinned $name" >&2
      return 1
    fi
    expect_version "$CLANG_EXPECT" "$actual" --version || return 1
  done
}

# check_llvm_cache: refuse unless the configured cache holds the pin's flags, the settings this script adds, and the
# pinned compilers.
check_llvm_cache() {
  local cache="$llvm_build/CMakeCache.txt"
  check_cache "$cache" "$LLVM_CMAKE_FLAGS" && check_cache "$cache" "$(llvm_settings | tr '\n' ' ')" \
    && check_compilers "$cache"
}

# configure_llvm: run the one configure of LLVM's llvm/ directory (README.md:89), with the pin's flags, the pinned
# compilers by name, and llvm_settings, and check its cache.
configure_llvm() {
  local -a flags settings
  read -r -a flags <<<"$LLVM_CMAKE_FLAGS"
  mapfile -t settings < <(llvm_settings)
  "$CMAKE" -S "$llvm_src/llvm" -B "$llvm_build" "${flags[@]}" -DCMAKE_C_COMPILER="$CLANG_C" \
    -DCMAKE_CXX_COMPILER="$CLANG_CXX" "${settings[@]}"
  check_llvm_cache
}

# build_targets: build the pin's TARGETS and nothing else, not the all or install targets, at LASSI_JOBS jobs.
build_targets() {
  local -a targets
  read -r -a targets <<<"$TARGETS"
  "$CMAKE" --build "$llvm_build" --target "${targets[@]}" --parallel "$LASSI_JOBS"
}

# record_fetched_sources: write lassi-fetched-sources.txt in the prefix, one line per source directory the configure
# fetched: FetchContent's (_deps/<name>-src) and ExternalProject's (<name>-prefix/src/<name>) in the build tree, where
# Polygeist's own fetches (Polymer's dependencies, off here) would go too. Each line is the directory's path in the
# prefix, then the git commit it holds and the count of tracked files that differ from it, or, for one unpacked from
# an archive, the sha256 of the sorted list of its files' sha256 lines. The one line none means nothing, which the
# spike expects (plans/spikes/p5-mlir-pins.md, Results 5: LLVM and Polygeist with Polymer off fetch nothing).
record_fetched_sources() {
  local out="$prefix/$fetched_name" found dir commit changed digest
  found="$(find "$llvm_build" -type d \( -path '*/_deps/*-src' -o -path '*-prefix/src/*' \) -prune \
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
  echo "polygeist: the configure's fetched sources, recorded in $out:"
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
    echo "polygeist: cannot compare the fetched sources: $tree or $tracked is missing" >&2
    return 1
  fi
  theirs="$(sed -e 's/[[:space:]]*$//' -e '/^#/d' -e '/^$/d' "$tracked")"
  if [ "$theirs" = PLACEHOLDER ]; then
    echo "polygeist: $tracked is PLACEHOLDER, no record; copy $tree into it and commit it first" >&2
    return 1
  fi
  ours="$(sed -e 's/[[:space:]]*$//' -e '/^#/d' -e '/^$/d' "$tree")"
  if [ "$ours" != "$theirs" ]; then
    echo "polygeist: $tree differs from $tracked; a change there is a pin change (Decision Log)" >&2
    diff <(printf '%s\n' "$theirs") <(printf '%s\n' "$ours") >&2 || true
    return 1
  fi
  echo "polygeist: $tree equals $tracked"
}

# check_tools <expected> <clang expected>: refuse unless CGEIST, POLYGEIST_OPT, MLIR_OPT, and MLIR_TRANSLATE are
# executable files in the prefix whose --version output holds <expected> (EXPECT_VERSION), LLVM_CLANG is one whose
# output holds <clang expected> (LLVM_CLANG_EXPECT), and clang's builtin headers lie in
# <bin>/../lib/clang/<LLVM major>/include, where cgeist finds them (polygeist.pin, CGEIST). Each tool's line that holds
# the expected text is printed; for that clang it is the whole first line, with the origin URL and HEAD.
check_tools() {
  local expected="$1" clang_expected="$2" key tool want out include
  for key in CGEIST POLYGEIST_OPT MLIR_OPT MLIR_TRANSLATE LLVM_CLANG; do
    tool="$prefix/${!key}"
    want="$expected"
    if [ "$key" = LLVM_CLANG ]; then want="$clang_expected"; fi
    if [ ! -f "$tool" ] || [ ! -x "$tool" ]; then
      echo "polygeist: $tool ($key) is not an executable file" >&2
      return 1
    fi
    if ! out="$("$tool" --version 2>&1)" || [[ "$out" != *"$want"* ]]; then
      echo "polygeist: $tool --version does not print '$want':" >&2
      printf '%s\n' "$out" >&2
      return 1
    fi
    echo "polygeist: $tool --version: $(printf '%s\n' "$out" | grep -F -m 1 -- "$want")"
  done
  include="$(dirname "$prefix/$LLVM_CLANG")/../lib/clang/${LLVM_VERSION%%.*}/include"
  if [ ! -f "$include/stddef.h" ]; then
    echo "polygeist: clang's builtin headers are not in $include, where cgeist looks for them" >&2
    return 1
  fi
  echo "polygeist: clang's builtin headers are in $include"
}

# verify_sources: refuse unless Polygeist (with LLVM_COMMIT as its gitlink) and llvm-project are at their pinned
# commits with no tracked file changed.
verify_sources() {
  verify_clean "$polygeist_src" "$COMMIT" && check_gitlink "$polygeist_src" && verify_clean "$llvm_src" "$LLVM_COMMIT"
}

# verify_install: the checks a kept install must pass: the sources, the configured cache, the configure's fetched
# sources against the tracked list (refused while that list is PLACEHOLDER), and the tools' versions and clang's
# builtin headers.
verify_install() {
  verify_sources && check_llvm_cache && check_fetched_sources "$prefix/$fetched_name" "$tracked_fetched" \
    && check_tools "$EXPECT_VERSION" "$LLVM_CLANG_EXPECT"
}

# check_root_fs <marker> <dir>...: fail when this user changed anything under the directories (the system
# temporary directories, where cgeist would put its /tmp file and Polygeist's PGO data would go by default, and /opt)
# since <marker>, without crossing into other file systems (Agent Rule 7). find's errors are kept and read:
# "Permission denied" lines are normal in shared directories and are counted, while a directory that is not there,
# any other error, or a failure with no message fails the check, so a find that could not look is never read as a
# clean one. find prints each hit as "changed <path>". The method of toolchains/tt-metal.sh (task P4.2).
check_root_fs() {
  local marker="$1" dir out line status=0 denied=0 leaked="" failed=""
  shift
  for dir in "$@"; do
    if [ ! -d "$dir" ]; then
      echo "polygeist: $dir is not a directory, so the root-filesystem check cannot run; nothing is recorded" >&2
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
    echo "polygeist: the root-filesystem check over $* did not run cleanly (find rc=$status); nothing is recorded" >&2
    printf '%s' "$failed" >&2
    return 1
  fi
  if [ -n "$leaked" ]; then
    echo "polygeist: files changed under $* during the install; nothing is recorded:" >&2
    printf '%s' "$leaked" >&2
    return 1
  fi
  echo "polygeist: root-filesystem check passed: nothing changed under $* since the start" \
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
    echo "polygeist: $prefix carries the pinned record but does not check out; it is left as it is (agents never" \
      "delete on the build host): inspect it, and ask the owner before anything there is moved or removed" >&2
    exit 1
  fi
  echo "polygeist: $prefix already holds the pinned Polygeist $COMMIT"
  exit 0
fi

echo "polygeist: started at $(date -u +%Y-%m-%dT%H:%M:%SZ) with $LASSI_JOBS jobs"
# The root-filesystem check below lists files changed after this marker.
marker="$(mktemp "$TMPDIR/polygeist-start.XXXXXX")"
# Every refusal comes before the first download: what is at the prefix, the host tools, and the space for both MLIR
# stacks together (plans/p5-ir.md, P5.4 and P5.5). Once tt-mlir's install is complete, du counts that stack and the
# plan is this stack's INSTALL_KIB alone; before it, the plan adds RESERVE_KIB, tt-mlir's stack, so whichever stack
# is built first leaves room for the other. On a resume the plan is the full figure again, though du already counts
# the unfinished tree, so its bytes are counted twice. That is kept on purpose as a fail-safe: on a resume it never
# plans less than a fresh start would. A resume near the stop line is therefore expected to be refused; queue it for
# the owner then.
check_prefix
check_host
if tt_mlir_installed; then
  check_space "$INSTALL_KIB"
else
  check_space "$((INSTALL_KIB + RESERVE_KIB))"
fi
if [ ! -e "$prefix" ]; then
  mkdir -p "$LASSI_TOOLCHAINS"
  mkdir "$prefix"
  pinned_record >"$prefix/$unfinished_name"
fi

# Polygeist first, and its gitlink checked, before llvm-project's larger fetch.
fetch_at "$polygeist_src" "$URL" "$COMMIT" --depth 1
verify_clean "$polygeist_src" "$COMMIT"
check_gitlink "$polygeist_src"
fetch_at "$llvm_src" "$LLVM_URL" "$LLVM_COMMIT" --depth 1
verify_clean "$llvm_src" "$LLVM_COMMIT"
configure_llvm
# The configure's fetches are recorded and, once the tracked list holds a record, compared before anything is built.
record_fetched_sources
if tracked_list_is_placeholder; then
  echo "polygeist: first install: $tracked_fetched is PLACEHOLDER, so nothing is compared now. Copy" \
    "$prefix/$fetched_name into it and commit it; until then every later run refuses this install."
else
  check_fetched_sources "$prefix/$fetched_name" "$tracked_fetched"
fi
build_targets
check_tools "$EXPECT_VERSION" "$LLVM_CLANG_EXPECT"
# The build changed no tracked file of the pinned sources.
verify_sources

# Nothing may land on the root filesystem (Agent Rule 7): the install is not recorded when this user changed
# anything in the system temporary directories or /opt since the run started, or when that cannot be checked.
check_root_fs "$marker" /tmp /var/tmp /opt

# Record: the unfinished marker, which holds the pinned record, becomes the install record.
mv -T "$prefix/$unfinished_name" "$prefix/$record_name"
echo "polygeist: installed $PREFIX_NAME at $COMMIT in $prefix"
echo "polygeist: finished at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
du -sk "$prefix"
du -sk "$LASSI_SCRATCH"
