"""Tests for the text-policy modes the hooks and the P0 gate rely on (P0.12).

tests/tools/test_text_policy.py covers the pattern, ASCII, branch, and
stdin checks. These tests cover the rest of the local half of the
enforcement (bible Attribution Policy, Enforcement items 1, 3, and 5):
`--message` (the commit-msg hook), `--history` (all commits, branches, and
the HEAD tree), and `tools/policy_canary.py local` (both seeded commits
blocked through the real hooks). History tests run in throwaway git
repositories under tmp_path with the commit identity given by environment
variables, so no git configuration is changed. The seeded text is the
checker's built-in canary token, so no listed string is ever written here.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CHECKER = REPO / "tools" / "check_text_policy.py"
sys.path.insert(0, str(REPO / "tools"))

import check_text_policy as ctp  # noqa: E402

IDENTITY = {
    "GIT_AUTHOR_NAME": "Policy Test",
    "GIT_AUTHOR_EMAIL": "policy-test@example.invalid",
    "GIT_COMMITTER_NAME": "Policy Test",
    "GIT_COMMITTER_EMAIL": "policy-test@example.invalid",
}


def checker(args: list[str], cwd: Path, patterns: Path) -> subprocess.CompletedProcess[str]:
    """Run the checker with a pattern file that holds one ordinary pattern; the canary token is always built in."""
    env = dict(os.environ)
    env.pop("TEXT_POLICY_PATTERNS", None)
    return subprocess.run(
        [sys.executable, str(CHECKER), "--patterns", str(patterns), *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def patterns(tmp_path: Path) -> Path:
    """Write a pattern file with one ordinary pattern and return its path."""
    path = tmp_path / "patterns.txt"
    path.write_text("(?i)\\bforbidden-vendor\\b\n", encoding="ascii")
    return path


def git(repo: Path, *args: str) -> None:
    """Run git in `repo` with the test identity from the environment; fail the test on an error."""
    env = {**os.environ, **IDENTITY}
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, env=env)


def new_repo(root: Path) -> Path:
    """Create an empty repository on branch main, with no hooks, and return its path."""
    repo = root / "repo"
    repo.mkdir()
    git(repo, "init", "--quiet", "-b", "main")
    git(repo, "config", "core.hooksPath", str(root / "no-hooks"))
    return repo


def commit(repo: Path, name: str, text: str, message: str) -> None:
    """Write `text` to `name` and commit it with `message`."""
    (repo / name).write_text(text, encoding="ascii")
    git(repo, "add", name)
    git(repo, "commit", "--quiet", "-m", message)


def test_message_mode_blocks_the_canary_and_passes_a_clean_message(tmp_path: Path, patterns: Path) -> None:
    bad = tmp_path / "bad-message.txt"
    bad.write_text(f"P0.12: Seed {ctp.canary_token()}\n", encoding="ascii")
    good = tmp_path / "good-message.txt"
    good.write_text("P0.12: Add a test\n\nA plain body.\n", encoding="ascii")
    blocked = checker(["--message", str(bad)], tmp_path, patterns)
    assert blocked.returncode == 1, blocked
    assert "commit message" in blocked.stdout + blocked.stderr
    assert checker(["--message", str(good)], tmp_path, patterns).returncode == 0


def test_message_mode_refuses_non_ascii(tmp_path: Path, patterns: Path) -> None:
    message = tmp_path / "message.txt"
    message.write_bytes(("P0.12: Add a caf" + chr(0xE9) + " test\n").encode("utf-8"))
    result = checker(["--message", str(message)], tmp_path, patterns)
    assert result.returncode == 1, result
    assert "[ascii]" in result.stdout + result.stderr


def test_history_passes_a_clean_repository(tmp_path: Path, patterns: Path) -> None:
    repo = new_repo(tmp_path)
    commit(repo, "README.md", "plain text\n", "P0.12: Add a readme")
    result = checker(["--history"], repo, patterns)
    assert result.returncode == 0, result


def test_history_catches_the_canary_in_a_committed_file(tmp_path: Path, patterns: Path) -> None:
    repo = new_repo(tmp_path)
    commit(repo, "README.md", "plain text\n", "P0.12: Add a readme")
    commit(repo, "notes.txt", f"{ctp.canary_token()}\n", "P0.12: Add notes")
    result = checker(["--history"], repo, patterns)
    assert result.returncode == 1, result
    assert "notes.txt" in result.stdout + result.stderr


def test_history_catches_the_canary_in_an_old_commit_message(tmp_path: Path, patterns: Path) -> None:
    repo = new_repo(tmp_path)
    commit(repo, "a.txt", "one\n", f"P0.12: Seed {ctp.canary_token()}")
    commit(repo, "b.txt", "two\n", "P0.12: A clean later commit")
    result = checker(["--history"], repo, patterns)
    assert result.returncode == 1, result
    assert "message" in result.stdout + result.stderr


def test_history_catches_a_listed_pattern_in_a_commit_message(tmp_path: Path, patterns: Path) -> None:
    repo = new_repo(tmp_path)
    commit(repo, "a.txt", "one\n", "P0.12: Mention Forbidden-Vendor")
    assert checker(["--history"], repo, patterns).returncode == 1


def test_canary_local_blocks_both_seeded_commits() -> None:
    result = subprocess.run(
        [sys.executable, str(REPO / "tools" / "policy_canary.py"), "local"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result
    report = json.loads(result.stdout.strip().splitlines()[0])
    assert report == {"content_commit_blocked": True, "message_commit_blocked": True}
    # Both blocks must come from real findings on the canary, not from a hook failing closed on its configuration.
    assert result.stdout.count("violations found") == 2, result.stdout
    assert "canary.txt" in result.stdout and "commit message" in result.stdout, result.stdout
