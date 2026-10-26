#!/usr/bin/env bash
# Compile assets/skeletons/tt-host-skeleton-v0/host.cpp exactly as the ttmetal-host toolchain compiles a TT host
# program (lassi/toolchains/ttmetal_build.py, TtMetalHost.command; toolchains/tt-metal.pin, the HOST_* keys):
#
#   <EXECUTABLE> HOST_DEFINES HOST_INCLUDES -idirafter . HOST_FLAGS HOST_LINK_FLAGS -o main host.cpp
#       HOST_LINK_LIBRARIES
#
# with {TREE} replaced by the pinned tree, $LASSI_TOOLCHAINS/<PREFIX_NAME>, run in a fresh build directory that holds
# host.cpp and the harness header lassi_io.h beside it, as build() writes them. Before the compile it checks what
# the stage runner checks (TtMetalHost.check_tree and the --version check): the install record names NAME and
# COMMIT, the tree's lassi-cpm-sources.txt equals toolchains/tt-metal-cpm-sources.txt, and the compiler's first
# --version line is EXPECT_VERSION. It prints the compile's exit status and the first 40 lines of its errors.
#
# For alpha01, from the repository root of a clean commit, through the remote runner, for example:
#   uv run tools/rx.py run -- 'bash <path of this script>'
# It writes only under $LASSI_SCRATCH (Agent Rule 7) and runs no program: the compiled host program is never run.
prlimit --pid $$ --core=1:1
set -euo pipefail

limit=$(prlimit --pid $$ --core --output SOFT,HARD --noheadings | tr -s ' ' | sed 's/^ //')
if [ "$limit" != "1 1" ]; then
    echo "skeleton_build: the core limit reads '$limit', not '1 1'; stopping" >&2
    exit 2
fi
echo "core limit (soft hard): $limit"

# The checkout is the working directory, where the remote runner starts a command.
repo=$PWD
skeleton="$repo/assets/skeletons/tt-host-skeleton-v0"
harness="$repo/assets/harness/c/lassi_io.h"
if [ ! -f "$repo/toolchains/tt-metal.pin" ] || [ ! -f "$skeleton/host.cpp" ]; then
    echo "skeleton_build: run from the repository root of a checkout that holds $skeleton; stopping" >&2
    exit 2
fi
head=$(git -C "$repo" rev-parse HEAD 2>/dev/null || echo unknown)
changed=$(git -C "$repo" status --porcelain 2>/dev/null | wc -l)
echo "commit: $head ($changed changed paths)"

# The pin: NAME, COMMIT, PREFIX_NAME, EXECUTABLE, EXPECT_VERSION, and the HOST_* words.
source "$repo/toolchains/tt-metal.pin"
tree="${LASSI_TOOLCHAINS:?LASSI_TOOLCHAINS is not set; run through the remote runner}/$PREFIX_NAME"
scratch="${LASSI_SCRATCH:?LASSI_SCRATCH is not set; run through the remote runner}"
echo "pin: $NAME $COMMIT; tree: $tree"

# The skeleton's host.cpp must be the bytes its MANIFEST.yaml pins.
listed=$(awk '/^- path: host.cpp$/ { found = 1; next } found && /sha256:/ { print $2; exit }' "$skeleton/MANIFEST.yaml")
actual=$(sha256sum "$skeleton/host.cpp" | cut -d' ' -f1)
if [ "$listed" != "$actual" ]; then
    echo "skeleton_build: host.cpp's sha256 $actual is not MANIFEST.yaml's $listed; stopping" >&2
    exit 2
fi
echo "host.cpp sha256: $actual"

# The install record's first line names NAME and COMMIT (lassi.toolchains.ttmetal_build check_install_record).
record=$(head -n 1 "$tree/lassi-install.txt" | awk '{ print $1, $2 }')
if [ "$record" != "$NAME $COMMIT" ]; then
    echo "skeleton_build: $tree/lassi-install.txt names '$record', not '$NAME $COMMIT'; stopping" >&2
    exit 2
fi

# The tree's CMake-fetched packages equal the tracked list, comments and blank lines left out (check_cpm_sources).
tracked=$(sed 's/[[:space:]]*$//' "$repo/toolchains/tt-metal-cpm-sources.txt" | grep -v '^#' | grep -v '^$' || true)
listed_cpm=$(sed 's/[[:space:]]*$//' "$tree/lassi-cpm-sources.txt" | grep -v '^$' || true)
if [ "$tracked" != "$listed_cpm" ]; then
    echo "skeleton_build: $tree/lassi-cpm-sources.txt differs from toolchains/tt-metal-cpm-sources.txt; stopping" >&2
    exit 2
fi

# The compiler is the pinned one (EXECUTABLE, EXPECT_VERSION).
version=$("$EXECUTABLE" --version | head -n 1)
if [ "$version" != "$EXPECT_VERSION" ]; then
    echo "skeleton_build: $EXECUTABLE --version reads '$version', not '$EXPECT_VERSION'; stopping" >&2
    exit 2
fi
echo "compiler: $EXECUTABLE ($version)"

# The HOST_* words with {TREE} replaced, split on spaces as host_words splits them.
read -r -a defines <<< "${HOST_DEFINES//"{TREE}"/$tree}"
read -r -a includes <<< "${HOST_INCLUDES//"{TREE}"/$tree}"
read -r -a flags <<< "${HOST_FLAGS//"{TREE}"/$tree}"
read -r -a link_flags <<< "${HOST_LINK_FLAGS//"{TREE}"/$tree}"
read -r -a link_libraries <<< "${HOST_LINK_LIBRARIES//"{TREE}"/$tree}"

# A fresh build directory under the scratch root, with host.cpp and lassi_io.h beside it, as build() writes them.
build=$(mktemp -d "$scratch/skeleton-build.XXXXXX")
cp "$skeleton/host.cpp" "$build/host.cpp"
cp "$harness" "$build/lassi_io.h"
mkdir -m 0700 "$build/tmp"
echo "build directory: $build"

command=("$EXECUTABLE" "${defines[@]}" "${includes[@]}" -idirafter . "${flags[@]}" "${link_flags[@]}" -o main host.cpp
         "${link_libraries[@]}")
echo "command: ${command[*]}"

# The compile, in the build directory, with only PATH and a private TMPDIR, as the compile sandbox gives it
# (lassi.executors.sandbox SandboxedCompileRunner), and the toolchain's 600 s timeout.
set +e
(cd "$build" && env -i PATH=/usr/sbin:/usr/bin:/sbin:/bin TMPDIR="$build/tmp" timeout 600 "${command[@]}") \
    > "$build/stdout.txt" 2> "$build/stderr.txt"
status=$?
set -e

echo "compile exit status: $status"
echo "artifact: $( [ -f "$build/main" ] && echo "main, $(stat -c %s "$build/main") bytes" || echo none )"
errors=$(grep -c -E '(^|: )(fatal )?error' "$build/stderr.txt" || true)
echo "error lines: $errors; warning lines: $(grep -c -E ': warning:' "$build/stderr.txt" || true)"
echo "---- first 40 lines of errors ----"
if [ "$errors" != "0" ]; then
    grep -E '(^|: )(fatal )?error' "$build/stderr.txt" | head -n 40
else
    head -n 40 "$build/stderr.txt"
fi
echo "---- end ----"
exit "$status"
