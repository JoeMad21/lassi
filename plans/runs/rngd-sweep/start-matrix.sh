#!/usr/bin/env bash
# Start the feature matrix (matrix.sh) on alpha01 from the workstation, in an owner's window, with one command, so a
# window's testing never waits on a session:
#   bash plans/runs/rngd-sweep/start-matrix.sh "<window end, as date -d reads it, e.g. 2026-10-25 04:00 EDT>"
# It starts the job from a clean worktree of the repository's committed HEAD, beside the repository at
# ../lassi-window-run (made on first use, reset to HEAD each time), so work in progress in the main tree never makes
# the run dirty. It puts the deadline 15 minutes before the window's end and starts one big job, in its own slot
# (rngd-matrix), whose rx timeout covers the time left. Agent Rule 17: run it only inside a window the owner
# approved. Poll with `uv run tools/rx.py job status`; pull the tables afterwards with
#   uv run tools/rx.py pull --path lassi-runs/rngd-sweep/tables6 --into <dir>
set -eu
end=${1:?usage: start-matrix.sh "<window end, e.g. 2026-10-25 04:00 EDT>"}
repo=$(cd "$(dirname "$0")/../../.." && pwd)
commit=$(git -C "$repo" rev-parse HEAD)
run_tree="$(dirname "$repo")/lassi-window-run"
if [ -d "$run_tree" ]; then
  git -C "$run_tree" checkout -q --detach --force "$commit"
  git -C "$run_tree" clean -qfd
else
  git -C "$repo" worktree add -q --detach "$run_tree" "$commit"
fi
[ -z "$(git -C "$run_tree" status --porcelain)" ] || { echo "start-matrix: $run_tree is not clean" >&2; exit 2; }
deadline=$(( $(date -d "$end" +%s) - 900 ))
left=$(( deadline - $(date +%s) ))
[ "$left" -gt 1800 ] || { echo "start-matrix: under 30 minutes before the deadline; not started" >&2; exit 2; }
echo "start-matrix: commit ${commit:0:12} from $run_tree; deadline $(date -d "@$deadline"), $left s left"
cd "$run_tree"
PYTHONIOENCODING=utf-8 uv run tools/rx.py job start --big --slot rngd-matrix --name rngd-matrix \
  --timeout $(( left + 900 )) -- "bash plans/runs/rngd-sweep/matrix.sh $deadline"
