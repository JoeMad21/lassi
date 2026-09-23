#!/usr/bin/env python3
"""Check that this clone is ready for work. Prints PASS, WARN, or FAIL per item.

  uv run tools/check_setup.py            local checks
  uv run tools/check_setup.py --remote   also run rx doctor against the build host

Exit status is 1 when any item FAILs. Standard library only.
"""

from __future__ import annotations

import argparse
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PATTERNS = Path.home() / ".config" / "lassi" / "text-policy.txt"
results = []


def out(cmd, cwd=REPO):
    p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return p.returncode, p.stdout.decode("utf-8", errors="replace").strip()


def record(level: str, item: str, detail: str = "") -> None:
    results.append(level)
    print(f"{level:4}  {item}" + (f": {detail}" if detail else ""))


def check_git() -> None:
    if not shutil.which("git"):
        record("FAIL", "git on PATH")
        return
    rc, _ = out(["git", "rev-parse", "--show-toplevel"])
    record("PASS" if rc == 0 else "FAIL", "inside a git repository")
    hooks = out(["git", "config", "--get", "core.hooksPath"])[1]
    record("PASS" if hooks == ".githooks" else "FAIL", "core.hooksPath is .githooks",
           "" if hooks == ".githooks" else "run: git config core.hooksPath .githooks")
    name, email = out(["git", "config", "user.name"])[1], out(["git", "config", "user.email"])[1]
    record("PASS" if name and email else "FAIL", "git identity set", f"{name} <{email}>")
    if platform.system() == "Windows":
        crlf = out(["git", "config", "--get", "core.autocrlf"])[1]
        record("PASS" if crlf == "false" else "WARN", "core.autocrlf false on Windows", crlf or "unset")
        lp = out(["git", "config", "--get", "core.longpaths"])[1]
        record("PASS" if lp == "true" else "WARN", "core.longpaths true on Windows", lp or "unset")
    branch = out(["git", "symbolic-ref", "--quiet", "--short", "HEAD"])[1]
    ok = bool(re.match(r"^(main|p\d+-[a-z0-9][a-z0-9-]*)$", branch))
    record("PASS" if ok else "WARN", "branch name follows main or p<phase>-<topic>", branch)
    remote = out(["git", "remote", "get-url", "origin"])[1]
    record("PASS" if "github.com" in remote else "WARN", "origin is a GitHub remote", remote or "not set")


def check_patterns() -> None:
    if not PATTERNS.is_file():
        record("FAIL", "pattern list present", f"missing {PATTERNS}")
        return
    lines = [ln.strip() for ln in PATTERNS.read_text(encoding="utf-8").splitlines()
             if ln.strip() and not ln.strip().startswith("#")]
    bad = []
    for ln in lines:
        try:
            re.compile(ln)
        except re.error:
            bad.append(ln)
    record("PASS" if lines and not bad else "FAIL", "pattern list compiles", f"{len(lines)} patterns")
    rc, text = out([sys.executable, str(REPO / "tools" / "check_text_policy.py"), "--self-test"])
    record("PASS" if rc == 0 else "FAIL", "text-policy self-test")
    rc, text = out([sys.executable, str(REPO / "tools" / "check_text_policy.py"), "--files",
                    "AGENTS.md", "docs/BIBLE.md", "README.md"])
    record("PASS" if rc == 0 else "FAIL", "committed agent docs pass the policy", text[-300:] if rc else "")


def check_tools() -> None:
    uv = shutil.which("uv")
    record("PASS" if uv else "FAIL", "uv on PATH", uv or "install from https://docs.astral.sh/uv/")
    if uv:
        rc, text = out(["uv", "python", "find", "3.10"])
        record("PASS" if rc == 0 else "WARN", "Python 3.10 available to uv", text if rc == 0 else
               "run: uv python install 3.10")
    record("PASS" if shutil.which("ssh") else "FAIL", "ssh on PATH")
    gh = shutil.which("gh")
    if gh:
        rc, _ = out(["gh", "auth", "status"])
        record("PASS" if rc == 0 else "WARN", "gh CLI authenticated")
    else:
        record("WARN", "gh CLI on PATH", "needed for CI evidence and pull requests")
    rc, text = out([sys.executable, str(REPO / "tools" / "status.py"), "check"])
    record("PASS" if rc == 0 else "FAIL", "plans/STATUS.md parses", text[-300:])


def check_remote() -> None:
    rc, text = out([sys.executable, str(REPO / "tools" / "rx.py"), "doctor"])
    record("PASS" if rc == 0 else "FAIL", "rx doctor", text.splitlines()[-1] if text else "")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--remote", action="store_true")
    a = p.parse_args()
    check_git()
    check_patterns()
    check_tools()
    if a.remote:
        check_remote()
    fails = results.count("FAIL")
    print(f"\n{fails} FAIL, {results.count('WARN')} WARN, {results.count('PASS')} PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
