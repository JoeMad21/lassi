"""End-to-end tests for tools/rx.py and tools/server/gate.py over the local transport."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture()
def env(tmp_path):
    repo = tmp_path / "repo"
    shutil.copytree(REPO, repo, ignore=shutil.ignore_patterns(".git", ".venv", ".rx", "__pycache__",
                                                               ".pytest_cache"))
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    # Drop the LASSI_* variables a gate sets (LASSI_SCRATCH above all) so the gate under test uses
    # this scratch, not the scratch of the gate that runs the suite on the build host.
    base = {k: v for k, v in os.environ.items() if not k.startswith("LASSI_")}
    e = dict(base, RX_TRANSPORT="local", RX_SCRATCH=str(scratch), RX_MACHINE="t",
             GIT_AUTHOR_NAME="T", GIT_AUTHOR_EMAIL="t@example.com", GIT_COMMITTER_NAME="T",
             GIT_COMMITTER_EMAIL="t@example.com")
    for cmd in (["git", "init", "-q", "-b", "main"], ["git", "add", "-A"], ["git", "commit", "-qm", "base"]):
        subprocess.run(cmd, cwd=repo, env=e, check=True, stdout=subprocess.DEVNULL)
    rx(repo, e, "bootstrap")
    return repo, e, scratch


def rx(repo, e, *args, check=True):
    p = subprocess.run([sys.executable, "tools/rx.py", *args], cwd=repo, env=e, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
    if check and p.returncode not in (0,):
        raise AssertionError(p.stdout.decode() + p.stderr.decode())
    return p.returncode, p.stdout.decode(), p.stderr.decode()


def test_run_clean_and_dirty(env):
    repo, e, scratch = env
    rc, out, _ = rx(repo, e, "run", "--", "echo hi-$LASSI_SLOT")
    assert rc == 0 and "hi-t-main" in out
    (repo / "extra.txt").write_text("x\n")
    rc, out, _ = rx(repo, e, "run", "--", "cat extra.txt")
    assert "snapshot" in out and "x" in out


def test_rc_propagates_and_provenance(env):
    repo, e, _ = env
    rc, out, _ = rx(repo, e, "run", "--", "exit 4", check=False)
    assert rc == 4
    run_id = out.strip().splitlines()[-1].split("id=")[1].split()[0]
    rx(repo, e, "pull", run_id, "--into", "results/t")
    meta = json.loads((repo / "results" / "t" / "provenance.json").read_text())
    assert meta["rc"] == 4 and meta["dirty"] is False and len(meta["commit"]) == 40


def test_job_lifecycle(env):
    repo, e, _ = env
    _, out, _ = rx(repo, e, "job", "start", "--name", "b", "--", "echo start; sleep 1; echo end")
    job = json.loads(out)["id"]
    rc, out, _ = rx(repo, e, "job", "wait", job, "--timeout", "30", "--interval", "1")
    assert rc == 0 and "end" in out


def test_refusals(env):
    repo, e, scratch = env
    assert rx(repo, e, "exec", "--", "tt-smi -r 0", check=False)[0] == 3
    assert rx(repo, e, "exec", "--", "sudo true", check=False)[0] == 3
    (scratch / "lassi-gate" / "STOP").touch()
    assert rx(repo, e, "run", "--", "true", check=False)[0] == 3
    (scratch / "lassi-gate" / "STOP").unlink()
    assert rx(repo, e, "pull", "--path", "../", check=False)[0] == 2


def test_only_wip_refs_accepted(env):
    repo, e, scratch = env
    p = subprocess.run(["git", "push", str(scratch / "lassi.git"), "HEAD:refs/heads/main"], cwd=repo, env=e,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert p.returncode != 0
