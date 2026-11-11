#!/usr/bin/env bash
# Move LASSI's build chain and tree from /mnt/nvme10/joseph_ufl into /mnt/nvme10/shrec-shared (owner's direction,
# 2026-11-11): the shared layer (hf, toolchains, venvs, cache) at the top, LASSI's own tree under lassi/.
#
#   bash shrec-move.sh          list what would move; change nothing
#   bash shrec-move.sh --do     move each entry (a rename on the same filesystem), leave a link at its old path
#
# Every move is `mv` then `ln -s <new> <old>`, so absolute paths baked into toolchains, venvs, worktrees, and old
# commits keep resolving. Nothing is deleted. The gate's own folder is not moved here (the owner moves it).
# Home files (dotfiles, .ssh, .local, .vscode-server, .config) and other projects stay where they are.
# MOVE_FROM and MOVE_TO override the two roots for tests.
set -u
J=${MOVE_FROM:-/mnt/nvme10/joseph_ufl}
S=${MOVE_TO:-/mnt/nvme10/shrec-shared}
L=$S/lassi
DO=0
[ "${1:-}" = --do ] && DO=1

fail() { echo "REFUSED: $*"; exit 2; }
[ -d "$J" ] && [ ! -L "$J" ] || fail "no source root $J"
[ -d "$S" ] || fail "no target root $S"
[ -w "$S" ] || fail "$S is not writable"
[ "$(stat -c %d "$J")" = "$(stat -c %d "$S")" ] || fail "$J and $S are on different filesystems; a move would copy"
busy=$(pgrep -u "$(id -u)" -f -a 'furiosa-llm serve|run-suite\.sh|run-passes\.sh|rngd-sweep/run\.sh|huggingface-cli download|hf download|snapshot_download|cmake --build|ninja' || true)
[ -z "$busy" ] || fail "a server, sweep, download, or build of this account is running:
$busy"

# source (relative to J) and target (absolute), in order; tmp last, since the running shell may use it
PLAN=(
  "toolchains|$S/toolchains"
  ".cache/huggingface|$S/hf"
  "hf/hub/models--Qwen--Qwen2.5-Coder-0.5B-Instruct|$S/hf/hub/models--Qwen--Qwen2.5-Coder-0.5B-Instruct"
  "furiosa-venv-2026.4|$S/venvs/furiosa-venv-2026.4"
  "ttrun-venv|$S/venvs/ttrun-venv"
  ".cache/uv|$S/cache/uv"
  ".cache/tt-metal-cache|$S/cache/tt-metal-cache"
  ".cache/furiosa|$S/cache/furiosa"
  ".cache/tiktoken|$S/cache/tiktoken"
  ".cache/lassi-p17-torchinductor|$S/cache/lassi-p17-torchinductor"
  ".cache/.furiosa-permit-probe|$S/cache/.furiosa-permit-probe"
  "lassi.git|$L/lassi.git"
  "lassi-wt|$L/lassi-wt"
  "lassi-runs|$L/lassi-runs"
  "lassi-live|$L/lassi-live"
  "bench|$L/bench"
  "rag|$L/rag"
  "train-data|$L/train-data"
  "downloads|$L/downloads"
  "tools|$L/tools"
  "sweep16-results-20261030-1526|$L/sweep16-results-20261030-1526"
  "sweep16-results-20261030-1526.tar.gz|$L/sweep16-results-20261030-1526.tar.gz"
  "sweep16-results-20261030-1529|$L/sweep16-results-20261030-1529"
  "sweep16-results-20261030-1529.tar.gz|$L/sweep16-results-20261030-1529.tar.gz"
  "sweep.sh|$L/sweep.sh"
  "skeleton-build.aegjXo|$L/skeleton-build.aegjXo"
  "skeleton-build.WsQkqK|$L/skeleton-build.WsQkqK"
  "all-blobs.txt|$L/all-blobs.txt"
  "kept-blobs.txt|$L/kept-blobs.txt"
  "orphan-blobs.txt|$L/orphan-blobs.txt"
  "bin|$L/bin"
  "tmp|$L/tmp"
)
# links inside the new layout, so scripts that build paths from the LASSI root keep working
COMPAT=(
  "$L/toolchains|$S/toolchains"
  "$L/hf|$S/hf"
  "$L/furiosa-venv-2026.4|$S/venvs/furiosa-venv-2026.4"
  "$L/ttrun-venv|$S/venvs/ttrun-venv"
  "$L/.cache/huggingface|$S/hf"
)

# A folder this script creates: readable by the shrec group, writable only by its owner, no inherited ACL.
newdir() {
  [ -d "$1" ] && return 0
  newdir "$(dirname "$1")"
  mkdir "$1" || fail "mkdir $1"
  if command -v setfacl >/dev/null; then setfacl -b "$1"; fi
  chmod 0755 "$1"
  echo "made $1"
}

echo "== $( [ $DO = 1 ] && echo moving || echo 'list only (pass --do to move)' ): $J -> $S"
abs=$(find "$J/.cache/huggingface" "$J/hf" -xdev -type l -lname '/*' 2>/dev/null | wc -l)
echo "absolute symlinks in the HF stores: $abs (they keep resolving through the links left behind)"
todo=0
for row in "${PLAN[@]}"; do
  src="$J/${row%%|*}"; dst="${row#*|}"
  if [ -L "$src" ] && [ "$(readlink "$src")" = "$dst" ]; then echo "done    $src"; continue; fi
  if [ ! -e "$src" ] && [ ! -L "$src" ]; then echo "absent  $src"; continue; fi
  [ -L "$src" ] && fail "$src is already a link to $(readlink "$src")"
  [ -e "$dst" ] || [ -L "$dst" ] && fail "$dst already exists"
  todo=$((todo + 1))
  if [ $DO = 0 ]; then echo "move    $src -> $dst"; continue; fi
  newdir "$(dirname "$dst")"
  mv -T "$src" "$dst" || fail "mv $src $dst"
  ln -s "$dst" "$src" || fail "ln -s $dst $src (the entry is at $dst; link it by hand)"
  echo "moved   $src -> $dst"
done
for row in "${COMPAT[@]}"; do
  link="${row%%|*}"; to="${row#*|}"
  if [ -L "$link" ] && [ "$(readlink "$link")" = "$to" ]; then echo "done    link $link"; continue; fi
  [ -e "$link" ] || [ -L "$link" ] && fail "$link exists and is not the expected link"
  if [ $DO = 0 ]; then echo "link    $link -> $to"; continue; fi
  newdir "$(dirname "$link")"
  ln -s "$to" "$link" || fail "ln -s $to $link"
  echo "linked  $link -> $to"
done
if [ $DO = 1 ] && [ -d "$L/lassi.git" ]; then
  slots=()
  for w in "$L"/lassi-wt/*/; do [ -f "$w/.git" ] && slots+=("${w%/}"); done
  for w in "${slots[@]}"; do  # point each slot at the repository's new path (the old one only resolves by link)
    line=$(head -1 "$w/.git")
    case "$line" in "gitdir: $J/lassi.git/"*) printf 'gitdir: %s\n' "$L/lassi.git/${line#"gitdir: $J/lassi.git/"}" > "$w/.git" ;; esac
  done
  if [ ${#slots[@]} -gt 0 ]; then
    git --git-dir="$L/lassi.git" worktree repair "${slots[@]}" 2>&1 | sed 's/^/repair: /'
    echo "worktrees repaired: ${#slots[@]}"
  fi
fi
echo "== $todo to move; free on the filesystem: $(df -h --output=avail "$S" | tail -1 | tr -d ' ')"
