"""Tests for tools/status.py."""

import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
import status  # noqa: E402


def fresh(tmp_path):
    dst = tmp_path / "STATUS.md"
    shutil.copy(REPO / "plans" / "STATUS.md", dst)
    return dst


def test_repo_status_parses():
    st = status.load()
    assert not st.errors


def test_first_item_is_p0_plan(tmp_path):
    st = status.load(fresh(tmp_path))
    assert status.next_item(st)["id"] == "P0.0"


def test_lifecycle_and_advance(tmp_path):
    f = fresh(tmp_path)
    run = lambda *a: status.main(["--file", str(f), *a])  # noqa: E731
    assert run("set", "P0.0", "DONE") == 0
    assert run("add", "P0.1", "Result record", "--depends", "P0.0") == 0
    assert run("add", "P0.G", "Phase gate", "--depends", "P0.1") == 0
    assert status.next_item(status.load(f))["id"] == "P0.1"
    assert run("set", "P0.1", "ACTIVE") == 0
    assert status.next_item(status.load(f))["resume"] is True
    assert run("set", "P0.1", "DONE") == 0
    assert run("set", "P0.G", "OWNER") == 0
    item = status.next_item(status.load(f))
    assert item["kind"] == "advance" and item["phase"] == "P1"


def test_single_active_enforced(tmp_path):
    f = fresh(tmp_path)
    status.main(["--file", str(f), "add", "P0.1", "a"])
    status.main(["--file", str(f), "set", "P0.0", "ACTIVE"])
    try:
        status.main(["--file", str(f), "set", "P0.1", "ACTIVE"])
    except SystemExit as exc:
        assert "already ACTIVE" in str(exc)
    else:
        raise AssertionError("second ACTIVE accepted")


def test_none_when_everything_waits(tmp_path):
    f = fresh(tmp_path)
    status.main(["--file", str(f), "set", "P0.0", "OWNER"])
    assert status.next_item(status.load(f))["kind"] == "none"
