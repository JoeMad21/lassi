#!/usr/bin/env bash
# Test shrec-move.sh on a fake tree; no host is contacted (task P17.63).
#   bash move-test.sh <work dir>     the fake tree goes in a new directory under <work dir>, removed on exit
# It lists, moves, and reruns the move (a rerun finds every entry done), then checks that moved files read through
# both paths, that home files and another project stay, that a git worktree works through its old and new paths and
# points at the new repository, and that the script refuses a target that exists and a sweep that is running.
set -u
D=$(cd "$(dirname "$0")" && pwd)
[ -d "${1:-}" ] || { echo "usage: bash move-test.sh <existing work dir>" >&2; exit 2; }
T=$(mktemp -d "$1/shrec-move-test.XXXX")
trap 'rm -rf "$T"' EXIT
J=$T/joseph_ufl
S=$T/shrec-shared
L=$S/lassi
mkdir -p "$S" "$J"/{toolchains/tt-metal@x/lib,.cache/uv,.cache/git,furiosa-venv-2026.4/bin,lassi-runs/x} \
  "$J"/{.cache/huggingface/hub/blobs,.cache/huggingface/hub/models--a--b/snapshots/r} \
  "$J"/{hf/hub/models--Qwen--Qwen2.5-Coder-0.5B-Instruct,bench,bin,tmp,.ssh,TurboQuant-Tenstorrent}
echo blob > "$J/.cache/huggingface/hub/blobs/h1"
ln -s ../../../blobs/h1 "$J/.cache/huggingface/hub/models--a--b/snapshots/r/w.bin"
printf '#!%s/furiosa-venv-2026.4/bin/python3\n' "$J" > "$J/furiosa-venv-2026.4/bin/furiosa-llm"
echo so > "$J/toolchains/tt-metal@x/lib/lib.so"
echo key > "$J/.ssh/id"
echo s > "$J/sweep.sh"
git init -q --bare "$J/lassi.git"
git clone -q "$J/lassi.git" "$T/c" 2>/dev/null
git -C "$T/c" -c user.name=t -c user.email=t@t commit -q --allow-empty -m t
git -C "$T/c" push -q origin HEAD:refs/heads/m
c=$(git -C "$T/c" rev-parse HEAD)
mkdir -p "$J/lassi-wt"
git --git-dir="$J/lassi.git" worktree add -q --detach "$J/lassi-wt/slot1" "$c"
run() { MOVE_FROM="$J" MOVE_TO="$S" bash "$D/shrec-move.sh" "$@"; echo "rc=$?"; }
short() { sed "s#$T#T#g"; }
echo "##### list"; run | short | tail -4
echo "##### move"; run --do | grep -v '^absent' | short | tail -24
echo "##### rerun: entries done"; run --do | grep -c '^done'
echo "##### checks"
test -L "$J/toolchains" && cat "$J/toolchains/tt-metal@x/lib/lib.so"
cat "$S/hf/hub/models--a--b/snapshots/r/w.bin" "$J/.cache/huggingface/hub/models--a--b/snapshots/r/w.bin"
cat "$L/.cache/huggingface/hub/blobs/h1"
test -d "$S/hf/hub/models--Qwen--Qwen2.5-Coder-0.5B-Instruct" && echo qwen-merged
test -x "$L/furiosa-venv-2026.4/bin" && echo venv-compat-ok
test -d "$J/.ssh" && ! test -L "$J/.ssh" && echo ssh-stays
test -d "$J/TurboQuant-Tenstorrent" && ! test -L "$J/TurboQuant-Tenstorrent" && echo other-project-stays
test -d "$J/.cache/git" && ! test -L "$J/.cache/git" && echo home-cache-stays
git -C "$L/lassi-wt/slot1" status --short >/dev/null && echo wt-new-path-ok
git -C "$J/lassi-wt/slot1" status --short >/dev/null && echo wt-old-path-ok
head -1 "$L/lassi-wt/slot1/.git" | short
echo "##### refuse: target exists"
mkdir -p "$J/rag" "$L/rag"; run | short | tail -2
rmdir "$L/rag"
echo "##### refuse: busy"
(exec -a "bash run-suite.sh" sleep 20) &
sleep 1; run | short | tail -3
kill %1 2>/dev/null
