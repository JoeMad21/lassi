"""Tests for tools/check_text_policy.py."""

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
import check_text_policy as ctp  # noqa: E402

PATS = [re.compile(re.escape(ctp.canary_token())), re.compile(r"(?i)\bforbidden-vendor\b"),
        re.compile(r"(?i)^co-authored-by:.*forbidden")]


def test_self_test_passes():
    assert ctp.self_test() == 0


def test_canary_not_in_source():
    assert ctp.canary_token() not in (REPO / "tools" / "check_text_policy.py").read_text()


def test_pattern_and_anchor_per_line():
    text = "P0.1: add loader\n\nCo-Authored-By: Forbidden Tool <x@y>\n"
    found = ctp.check_message_text(text, "msg", PATS)
    assert [v.rule for v in found] == ["pattern"]


def test_ascii_rule_and_exemption():
    assert ctp.scan_blob("docs/a.md", "caf\u00e9\n".encode(), PATS)
    assert not ctp.scan_blob("third_party/LASSI/a.py", "caf\u00e9\n".encode(), PATS)


def test_path_is_scanned():
    assert ctp.scan_blob("notes/forbidden-vendor.md", b"clean\n", PATS)


def test_branch_rule():
    assert not ctp.check_branch("p12-frontends", PATS)
    assert ctp.check_branch("tool/feature", PATS)


def test_missing_patterns_fail_closed(tmp_path, monkeypatch):
    monkeypatch.delenv("TEXT_POLICY_PATTERNS", raising=False)
    monkeypatch.setenv("LASSI_TEXT_POLICY_FILE", str(tmp_path / "missing.txt"))
    p = subprocess.run([sys.executable, str(REPO / "tools" / "check_text_policy.py"), "--text-stdin"],
                       input=b"hello", stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert p.returncode == 2


def test_cli_blocks_canary(tmp_path, monkeypatch):
    pf = tmp_path / "p.txt"
    pf.write_text("(?i)\\bforbidden-vendor\\b\n")
    monkeypatch.delenv("TEXT_POLICY_PATTERNS", raising=False)
    p = subprocess.run([sys.executable, str(REPO / "tools" / "check_text_policy.py"), "--patterns", str(pf),
                        "--text-stdin"], input=ctp.canary_token().encode(), stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
    assert p.returncode == 1
