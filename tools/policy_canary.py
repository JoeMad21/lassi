#!/usr/bin/env python3
"""Seed text-policy violations with the canary token to prove the checks block them.

The canary token is built into tools/check_text_policy.py, so no listed string is ever
committed or pushed. Commands:

  local    Create a throwaway branch in a temporary clone, try to commit the canary through
           the normal hooks, and report whether the commit was blocked (expected: blocked).
  push     Build a commit carrying the canary with git plumbing (hooks cannot run on plumbing)
           and push it to origin as branch p0-canary, so CI must fail on it.
  status   Show the CI conclusion for p0-canary (needs the gh CLI).
  cleanup  Delete the p0-canary branch on origin.

Evidence for the P0 gate: the 'local' output and the failed CI run URL from 'status'.
Standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from check_text_policy import canary_token  # noqa: E402

BRANCH = "p0-canary"


def run(args, cwd=REPO, check=True, env=None):
    p = subprocess.run(args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    out = p.stdout.decode("utf-8", errors="replace")
    if check and p.returncode != 0:
        raise SystemExit(f"canary: {' '.join(args)} failed:\n{out}")
    return p.returncode, out


def cmd_local() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        clone = Path(tmp) / "clone"
        run(["git", "clone", "--quiet", "--no-hardlinks", str(REPO), str(clone)])
        run(["git", "config", "core.hooksPath", ".githooks"], cwd=clone)
        run(["git", "checkout", "--quiet", "-b", BRANCH], cwd=clone)
        (clone / "canary.txt").write_text(f"seeded {canary_token()}\n")
        run(["git", "add", "canary.txt"], cwd=clone)
        rc, out = run(["git", "commit", "-m", "P0: canary content"], cwd=clone, check=False)
        content_blocked = rc != 0
        (clone / "canary.txt").write_text("clean\n")
        run(["git", "add", "canary.txt"], cwd=clone)
        rc2, out2 = run(["git", "commit", "-m", f"P0: canary message {canary_token()}"], cwd=clone, check=False)
        message_blocked = rc2 != 0
    print(json.dumps({"content_commit_blocked": content_blocked, "message_commit_blocked": message_blocked}))
    print(out.strip()[-600:])
    print(out2.strip()[-600:])
    return 0 if content_blocked and message_blocked else 1


def cmd_push() -> int:
    head = run(["git", "rev-parse", "HEAD"])[1].strip()
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ, GIT_INDEX_FILE=os.path.join(tmp, "index"))
        run(["git", "read-tree", head], env=env)
        blob = subprocess.run(["git", "hash-object", "-w", "--stdin"], cwd=REPO, input=f"{canary_token()}\n".encode(),
                              stdout=subprocess.PIPE, check=True).stdout.decode().strip()
        run(["git", "update-index", "--add", "--cacheinfo", f"100644,{blob},canary.txt"], env=env)
        tree = run(["git", "write-tree"], env=env)[1].strip()
        commit = run(["git", "commit-tree", tree, "-p", head, "-m", f"P0: CI canary {canary_token()}"])[1].strip()
    rc, out = run(["git", "push", "--no-verify", "--force", "origin", f"{commit}:refs/heads/{BRANCH}"], check=False)
    print(out.strip())
    print(json.dumps({"pushed": rc == 0, "commit": commit, "branch": BRANCH}))
    return rc


def cmd_status() -> int:
    if not shutil.which("gh"):
        print("canary: gh CLI not found; check the Actions tab for branch p0-canary")
        return 2
    rc, out = run(["gh", "run", "list", "--branch", BRANCH, "--limit", "3", "--json",
                   "databaseId,conclusion,status,url,headSha"], check=False)
    print(out.strip())
    return rc


def cmd_cleanup() -> int:
    rc, out = run(["git", "push", "--no-verify", "origin", "--delete", BRANCH], check=False)
    print(out.strip())
    return rc


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("command", choices=["local", "push", "status", "cleanup"])
    a = p.parse_args()
    return {"local": cmd_local, "push": cmd_push, "status": cmd_status, "cleanup": cmd_cleanup}[a.command]()


if __name__ == "__main__":
    sys.exit(main())
