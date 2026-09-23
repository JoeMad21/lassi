#!/usr/bin/env python3
"""Repository text-policy checker.

Enforces three rules on every repository surface:

1. Pattern rule: no line matches a pattern from the owner's pattern list.
   The list lives outside git. Local hooks read ``~/.config/lassi/text-policy.txt``
   (override with ``LASSI_TEXT_POLICY_FILE``); CI reads the ``TEXT_POLICY_PATTERNS``
   environment variable, filled from a repository Actions variable. One regex per
   line; blank lines and lines starting with ``#`` are ignored. A built-in canary
   pattern is always active so gates can seed a violation without committing any
   listed string (``--print-canary`` prints it).
2. ASCII rule: text in repository artifacts is plain ASCII (``third_party/`` exempt).
3. Branch rule: branch names are ``main`` or ``p<phase>-<topic>``.

Exit status: 0 clean, 1 violations, 2 configuration error. Standard library only.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

DEFAULT_PATTERN_FILE = Path.home() / ".config" / "lassi" / "text-policy.txt"
ASCII_EXEMPT_PREFIXES = ("third_party/",)
BINARY_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".ico", ".zip", ".gz", ".tgz", ".xz",
    ".bz2", ".parquet", ".npy", ".npz", ".bin", ".so", ".a", ".o", ".exe", ".dll",
    ".whl", ".pt", ".safetensors", ".onnx", ".pkl", ".woff", ".woff2", ".ttf",
}
BRANCH_RE = re.compile(r"^(main|p\d+-[a-z0-9][a-z0-9-]*)$")
ZERO_SHA = "0" * 40
MAX_REPORT = 200


def canary_token() -> str:
    """Return the canary token. Built from parts so this file never contains it."""
    return "-".join(["LASSI", "POLICY", "CANARY"])


@dataclass
class Violation:
    """One policy violation at a location."""

    where: str
    rule: str
    detail: str

    def render(self) -> str:
        return f"{self.where}: [{self.rule}] {self.detail}"


class ConfigError(Exception):
    """Raised when the pattern list is missing or invalid."""


def load_patterns(pattern_file: Optional[str], allow_missing: bool = False) -> List[re.Pattern]:
    """Load the external pattern list plus the canary pattern."""
    text: Optional[str] = None
    env_inline = os.environ.get("TEXT_POLICY_PATTERNS", "")
    if env_inline.strip():
        text = env_inline
    else:
        path = Path(pattern_file or os.environ.get("LASSI_TEXT_POLICY_FILE", "") or DEFAULT_PATTERN_FILE)
        if path.is_file():
            text = path.read_text(encoding="utf-8")
        elif not allow_missing:
            raise ConfigError(
                f"pattern list not found at {path}. Install it (see the local setup guide) "
                "or set TEXT_POLICY_PATTERNS. Refusing to pass without it."
            )
    patterns: List[re.Pattern] = [re.compile(re.escape(canary_token()))]
    if text is None:
        return patterns
    count = 0
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            patterns.append(re.compile(line))
            count += 1
        except re.error as exc:
            raise ConfigError(f"pattern line {lineno} does not compile: {exc}") from exc
    if count == 0 and not allow_missing:
        raise ConfigError("pattern list is empty; refusing to pass without patterns")
    return patterns


def scan_text(text: str, where: str, patterns: Sequence[re.Pattern], check_ascii: bool) -> List[Violation]:
    """Scan one text blob line by line for pattern and ASCII violations."""
    found: List[Violation] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for pat in patterns:
            m = pat.search(line)
            if m:
                detail = f"matches /{pat.pattern}/ at '{m.group(0)}'"
                found.append(Violation(f"{where}:{lineno}", "pattern", detail))
                break
        if check_ascii:
            for col, ch in enumerate(line, 1):
                if ord(ch) > 127:
                    found.append(Violation(f"{where}:{lineno}:{col}", "ascii", f"non-ASCII character U+{ord(ch):04X}"))
                    break
    return found


def is_binary(path: str, data: bytes) -> bool:
    """Heuristic binary detection by suffix or NUL byte."""
    return Path(path).suffix.lower() in BINARY_SUFFIXES or b"\x00" in data[:8192]


def ascii_applies(path: str) -> bool:
    norm = path.replace("\\", "/")
    return not norm.startswith(ASCII_EXEMPT_PREFIXES)


def scan_blob(path: str, data: bytes, patterns: Sequence[re.Pattern]) -> List[Violation]:
    """Scan a file path and its content."""
    found = scan_text(path, f"path {path}", patterns, check_ascii=True)
    if is_binary(path, data):
        return found
    text = data.decode("utf-8", errors="replace")
    found.extend(scan_text(text, path, patterns, check_ascii=ascii_applies(path)))
    return found


def git(*args: str, check: bool = True) -> str:
    out = subprocess.run(["git", *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and out.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {out.stderr.decode(errors='replace').strip()}")
    return out.stdout.decode("utf-8", errors="replace")


def git_bytes(*args: str) -> bytes:
    out = subprocess.run(["git", *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if out.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {out.stderr.decode(errors='replace').strip()}")
    return out.stdout


def check_message_text(text: str, where: str, patterns: Sequence[re.Pattern]) -> List[Violation]:
    body = "\n".join(line for line in text.splitlines() if not line.startswith("#"))
    return scan_text(body, where, patterns, check_ascii=True)


def check_branch(name: str, patterns: Sequence[re.Pattern]) -> List[Violation]:
    found = scan_text(name, f"branch {name}", patterns, check_ascii=True)
    if not BRANCH_RE.match(name):
        rule = "must be main or p<phase>-<topic> (lowercase, digits, dashes)"
        found.append(Violation(f"branch {name}", "branch", rule))
    return found


def check_staged(patterns: Sequence[re.Pattern]) -> List[Violation]:
    names = git("diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z").split("\0")
    found: List[Violation] = []
    for name in filter(None, names):
        found.extend(scan_blob(name, git_bytes("show", f":{name}"), patterns))
    return found


def check_files(paths: Iterable[str], patterns: Sequence[re.Pattern]) -> List[Violation]:
    found: List[Violation] = []
    for p in paths:
        fp = Path(p)
        if fp.is_file():
            found.extend(scan_blob(p.replace("\\", "/"), fp.read_bytes(), patterns))
    return found


def commits_in(rev_args: Sequence[str]) -> List[str]:
    out = git("rev-list", *rev_args, check=False)
    return [c for c in out.split() if c]


def check_commits(commits: Iterable[str], patterns: Sequence[re.Pattern]) -> List[Violation]:
    """Check messages and identities of commits."""
    found: List[Violation] = []
    for sha in commits:
        meta = git("show", "-s", "--format=%an <%ae>%n%cn <%ce>%n%B", sha)
        lines = meta.split("\n", 2)
        author, committer = lines[0], lines[1]
        body = lines[2] if len(lines) > 2 else ""
        short = sha[:10]
        found.extend(scan_text(body, f"commit {short} message", patterns, check_ascii=True))
        for label, ident in (("author", author), ("committer", committer)):
            found.extend(scan_text(ident, f"commit {short} {label}", patterns, check_ascii=True))
            if "[bot]" in ident.lower():
                found.append(Violation(f"commit {short} {label}", "identity", f"bot identity '{ident}'"))
    return found


def changed_files(base: Optional[str], head: str) -> List[str]:
    if base and base != ZERO_SHA:
        out = git("diff", "--name-only", "--diff-filter=ACMR", "-z", f"{base}...{head}", check=False)
    else:
        out = git("ls-tree", "-r", "--name-only", "-z", head)
    return [n for n in out.split("\0") if n]


def check_tree_files(names: Iterable[str], rev: str, patterns: Sequence[re.Pattern]) -> List[Violation]:
    found: List[Violation] = []
    for name in names:
        try:
            data = git_bytes("show", f"{rev}:{name}")
        except RuntimeError:
            continue
        found.extend(scan_blob(name, data, patterns))
    return found


def ci_range() -> tuple:
    """Return (base, head, commits) for the CI event from environment variables."""
    event = os.environ.get("EVENT_NAME", "push")
    if event == "pull_request":
        base = os.environ.get("BASE_SHA", "")
        head = os.environ.get("HEAD_SHA", "") or "HEAD"
        return base, head, commits_in([f"{base}..{head}"]) if base else commits_in([head])
    before = os.environ.get("BEFORE_SHA", "")
    head = os.environ.get("AFTER_SHA", "") or "HEAD"
    if before and before != ZERO_SHA:
        return before, head, commits_in([f"{before}..{head}"])
    has_main = git("rev-parse", "--verify", "--quiet", "refs/remotes/origin/main", check=False).strip()
    if has_main and os.environ.get("REF_NAME", "") != "main":
        return "origin/main", head, commits_in([head, "--not", "origin/main"])
    return None, head, commits_in([head])


def run_ci(patterns: Sequence[re.Pattern]) -> List[Violation]:
    if not os.environ.get("TEXT_POLICY_PATTERNS", "").strip():
        raise ConfigError("TEXT_POLICY_PATTERNS is empty; set the repository Actions variable")
    after = os.environ.get("AFTER_SHA", "")
    if after == ZERO_SHA:
        print("branch deletion; nothing to check")
        return []
    base, head, commits = ci_range()
    files = changed_files(base, head)
    found = check_commits(commits, patterns)
    found.extend(check_tree_files(files, head, patterns))
    ref = os.environ.get("REF_NAME", "")
    if ref:
        found.extend(check_branch(ref, patterns))
    for label in ("PR_TITLE", "PR_BODY"):
        found.extend(scan_text(os.environ.get(label, ""), label.lower(), patterns, check_ascii=True))
    print(f"checked {len(commits)} commits, {len(files)} files, ref '{ref}'")
    return found


def run_history(patterns: Sequence[re.Pattern]) -> List[Violation]:
    found = check_commits(commits_in(["--all"]), patterns)
    for ref in git("for-each-ref", "--format=%(refname:short)", "refs/heads").split():
        found.extend(check_branch(ref, patterns))
    found.extend(check_tree_files(changed_files(None, "HEAD"), "HEAD", patterns))
    return found


def self_test() -> int:
    """Exercise the rules with the canary and synthetic patterns."""
    pats = [re.compile(re.escape(canary_token())), re.compile(r"(?i)\bforbidden-vendor\b")]
    cases = [
        (bool(scan_text(f"x {canary_token()} y", "t", pats, True)), True, "canary detected"),
        (bool(scan_text("Forbidden-Vendor wrote this", "t", pats, True)), True, "pattern detected"),
        (bool(scan_text("plain ascii text", "t", pats, True)), False, "clean text passes"),
        (bool(scan_text("caf\u00e9", "t", pats, True)), True, "non-ASCII detected"),
        (bool(scan_text("caf\u00e9", "t", pats, False)), False, "ASCII exemption honored"),
        (bool(check_branch("p0-core", pats)), False, "valid branch passes"),
        (bool(check_branch("feature/x", pats)), True, "invalid branch fails"),
        (ascii_applies("third_party/LASSI/a.py"), False, "third_party exempt from ASCII"),
    ]
    bad = [name for got, want, name in cases if got != want]
    for got, want, name in cases:
        print(("ok   " if got == want else "FAIL ") + name)
    return 1 if bad else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--patterns", help="pattern file (default ~/.config/lassi/text-policy.txt)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--message", metavar="FILE", help="check a commit message file")
    g.add_argument("--staged", action="store_true", help="check staged paths and content")
    g.add_argument("--files", nargs="+", metavar="PATH", help="check files on disk")
    g.add_argument("--text-stdin", action="store_true", help="check text from stdin")
    g.add_argument("--branch", metavar="NAME", help="check a branch name")
    g.add_argument("--range", metavar="REVS", help="check commits from rev-list arguments, e.g. 'A..B'")
    g.add_argument("--history", action="store_true", help="check all commits, branches, HEAD tree")
    g.add_argument("--ci", action="store_true", help="CI mode; reads event data from the environment")
    g.add_argument("--self-test", action="store_true", help="run built-in rule tests")
    g.add_argument("--print-canary", action="store_true", help="print the canary token")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return self_test()
    if args.print_canary:
        print(canary_token())
        return 0
    try:
        patterns = load_patterns(args.patterns)
        if args.message:
            found = check_message_text(Path(args.message).read_text(encoding="utf-8", errors="replace"),
                                       "commit message", patterns)
        elif args.staged:
            found = check_staged(patterns)
        elif args.files:
            found = check_files(args.files, patterns)
        elif args.text_stdin:
            found = scan_text(sys.stdin.read(), "stdin", patterns, check_ascii=True)
        elif args.branch:
            found = check_branch(args.branch, patterns)
        elif args.range:
            found = check_commits(commits_in(args.range.split()), patterns)
        elif args.history:
            found = run_history(patterns)
        else:
            found = run_ci(patterns)
    except ConfigError as exc:
        print(f"text-policy: configuration error: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"text-policy: {exc}", file=sys.stderr)
        return 2
    if not found:
        return 0
    print("text-policy: violations found:", file=sys.stderr)
    for v in found[:MAX_REPORT]:
        print("  " + v.render(), file=sys.stderr)
    if len(found) > MAX_REPORT:
        print(f"  ... and {len(found) - MAX_REPORT} more", file=sys.stderr)
    print("Fix the text (do not bypass the check). False positive: record it in plans/OWNER-QUEUE.md.",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
