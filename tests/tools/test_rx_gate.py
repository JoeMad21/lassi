"""Tests for tools/rx.py and tools/server/gate.py.

Most run end to end over the local transport. The shared-root tests (task P17.63) cover the
gate's optional owner-edited config key "shared_root": the run environment and the doctor
verb with and without it, the refusal of a relative or filesystem-root value, rx's default
scratch root, and the default deny patterns for the shared roots on the build host.
"""

import importlib.util
import json
import os
import re
import shlex
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


# ---------------------------------------------------------------- shared root (task P17.63)

ABSENT = object()
SHARED_DEFAULT_SCRATCH = "/mnt/nvme10/shrec-shared/lassi"
ENV_NAMES = ["LASSI_SCRATCH", "LASSI_RUNS_ROOT", "LASSI_TOOLCHAINS", "LASSI_SHARED", "LASSI_VENVS", "TMPDIR",
             "XDG_CACHE_HOME", "UV_CACHE_DIR", "PIP_CACHE_DIR", "HF_HOME", "TIKTOKEN_CACHE_DIR", "CCACHE_DIR",
             "CARGO_HOME", "RUSTUP_HOME", "DOTNET_CLI_HOME", "NUGET_PACKAGES", "npm_config_cache"]
PROBE = ("import json, os\n"
         "env = {k: os.environ.get(k) for k in %r}\n"
         "env['PATH_HEAD'] = os.environ.get('PATH', '').split(os.pathsep)[0]\n"
         "print('RXENV=' + json.dumps(env))\n")


def set_shared_root(scratch, value):
    """Edit the bootstrapped gate's config.json as the owner would: set, null, or drop shared_root."""
    path = scratch / "lassi-gate" / "config.json"
    cfg = json.loads(path.read_text())
    if value is ABSENT:
        cfg.pop("shared_root", None)
    else:
        cfg["shared_root"] = value
    path.write_text(json.dumps(cfg, indent=2))


def norm(path):
    """Compare paths across Windows, Git Bash (/c/...), and Linux spellings."""
    text = str(path).replace("\\", "/")
    drive = re.match(r"^/([A-Za-z])(/.*)?$", text)
    if os.name == "nt" and drive:
        text = drive.group(1) + ":" + (drive.group(2) or "/")
    return os.path.normcase(os.path.normpath(text))


def run_env(repo, e):
    """Run a probe through `rx run` and return the variables the gate's environment gave it."""
    (repo / "env_probe.py").write_text(PROBE % (ENV_NAMES,))
    py = shlex.quote(sys.executable.replace("\\", "/"))
    _, out, _ = rx(repo, e, "run", "--", py + " env_probe.py")
    lines = [ln for ln in out.splitlines() if ln.startswith("RXENV=")]
    assert lines, out
    return json.loads(lines[-1][len("RXENV="):])


def mismatches(got, expected):
    """Return one line per variable whose value differs from the expected path."""
    bad = []
    for key, want in sorted(expected.items()):
        have = got.get(key)
        if have is None or norm(have) != norm(want):
            bad.append("%s=%s (want %s)" % (key, have, want))
    return bad


def scratch_layout(s):
    """Variables that stay under the scratch root S with or without a shared root."""
    return {"LASSI_SCRATCH": s, "LASSI_RUNS_ROOT": s / "lassi-runs", "TMPDIR": s / "tmp", "PATH_HEAD": s / "bin",
            "DOTNET_CLI_HOME": s / ".dotnet-home", "NUGET_PACKAGES": s / ".nuget" / "packages"}


def shared_layout(d):
    """Variables that move to the shared root D when config.json sets shared_root."""
    return {"LASSI_TOOLCHAINS": d / "toolchains", "HF_HOME": d / "hf", "XDG_CACHE_HOME": d / "cache",
            "UV_CACHE_DIR": d / "cache" / "uv", "PIP_CACHE_DIR": d / "cache" / "pip",
            "TIKTOKEN_CACHE_DIR": d / "cache" / "tiktoken", "CCACHE_DIR": d / "cache" / "ccache",
            "npm_config_cache": d / "cache" / "npm", "CARGO_HOME": d / "cargo", "RUSTUP_HOME": d / "rustup",
            "LASSI_SHARED": d, "LASSI_VENVS": d / "venvs"}


def legacy_layout(s):
    """Today's values when no shared root is set; LASSI_SHARED and LASSI_VENVS fall back to S."""
    return {"LASSI_TOOLCHAINS": s / "toolchains", "HF_HOME": s / "hf", "XDG_CACHE_HOME": s / ".cache",
            "UV_CACHE_DIR": s / ".cache" / "uv", "PIP_CACHE_DIR": s / ".cache" / "pip",
            "TIKTOKEN_CACHE_DIR": s / ".cache" / "tiktoken", "CCACHE_DIR": s / ".cache" / "ccache",
            "npm_config_cache": s / ".cache" / "npm", "CARGO_HOME": s / ".cargo", "RUSTUP_HOME": s / ".rustup",
            "LASSI_SHARED": s, "LASSI_VENVS": s}


def doctor_json(repo, e):
    """Return the gate's doctor result as `rx doctor` prints it (an indented JSON block)."""
    _, out, _ = rx(repo, e, "doctor")
    lines = out.splitlines()
    start = lines.index("{")
    return json.loads("\n".join(lines[start:lines.index("}", start) + 1]))


def deny_patterns():
    """The default deny patterns, compiled as the gate applies them (re.search over the command)."""
    cfg = json.loads((REPO / "tools" / "server" / "config.default.json").read_text())
    return [re.compile(p) for p in cfg["deny_patterns"]]


def test_run_env_with_shared_root(env, tmp_path):
    """With shared_root D, caches, toolchains, and venvs move to D; run state stays under S."""
    repo, e, scratch = env
    shared = tmp_path / "shared"
    shared.mkdir()
    set_shared_root(scratch, str(shared))
    got = run_env(repo, e)
    bad = mismatches(got, dict(scratch_layout(scratch), **shared_layout(shared)))
    assert not bad, "variables off the shared layout:" + "".join("\n  " + b for b in bad)


@pytest.mark.parametrize("value", [ABSENT, None], ids=["absent", "null"])
def test_run_env_without_shared_root(env, value):
    """Without shared_root, every existing variable keeps its value; LASSI_SHARED and LASSI_VENVS are S."""
    repo, e, scratch = env
    set_shared_root(scratch, value)
    got = run_env(repo, e)
    bad = mismatches(got, dict(scratch_layout(scratch), **legacy_layout(scratch)))
    assert not bad, "variables off today's layout:" + "".join("\n  " + b for b in bad)


def test_doctor_reports_shared_root(env, tmp_path):
    """Doctor reports shared (D when set, S when unset) and shared_free_gb as a number."""
    repo, e, scratch = env
    shared = tmp_path / "shared"
    shared.mkdir()
    for value, want in ((str(shared), shared), (ABSENT, scratch)):
        set_shared_root(scratch, value)
        res = doctor_json(repo, e)
        assert "shared" in res and norm(res["shared"]) == norm(want), res
        free = res.get("shared_free_gb")
        assert isinstance(free, (int, float)) and not isinstance(free, bool), res


@pytest.mark.parametrize("value", ["rel/shared", "/"], ids=["relative", "filesystem-root"])
def test_run_refuses_a_bad_shared_root(env, tmp_path, value):
    """A relative or filesystem-root shared_root fails `rx run` with a message naming shared_root; nothing runs."""
    repo, e, scratch = env
    set_shared_root(scratch, value)
    marker = tmp_path / "ran.txt"
    # The command prints ran-42 and writes the marker; its own text never holds "ran-42", so no echo of it matches.
    cmd = "echo ran-$((6*7)) && echo ran > " + shlex.quote(marker.as_posix())
    rc, out, err = rx(repo, e, "run", "--", cmd, check=False)
    assert rc != 0, out + err
    assert "shared_root" in out + err, out + err
    assert "ran-42" not in out + err and not marker.exists(), "the command ran:\n" + out + err


def test_rx_default_scratch_is_shared_lassi(tmp_path, monkeypatch):
    """With no RX_SCRATCH and no .rx.json, rx's scratch root is the shared tree's lassi directory."""
    for key in list(os.environ):
        if key.upper().startswith("RX_"):
            monkeypatch.delenv(key)
    spec = importlib.util.spec_from_file_location("rx_settings_under_test", REPO / "tools" / "rx.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "REPO", tmp_path)  # an empty directory: no .rx.json override
    assert not (tmp_path / ".rx.json").exists()
    assert mod.settings()["scratch"] == SHARED_DEFAULT_SCRATCH


@pytest.mark.parametrize("cmd", ["rm -rf /mnt/nvme10/joseph_ufl", "rm -rf /mnt/nvme10/shrec-shared",
                                 "rm -rf /mnt/nvme10/shrec-shared/", "rm -rf /mnt/nvme10/shrec-shared/lassi"])
def test_default_deny_patterns_refuse_root_deletes(cmd):
    """The default deny patterns refuse deleting the old scratch root, the shared root, and its lassi tree."""
    assert [p.pattern for p in deny_patterns() if p.search(cmd)] != []


def test_default_deny_patterns_allow_deletes_below_roots():
    """A delete below the roots is not refused by any default deny pattern."""
    cmd = "rm -rf /mnt/nvme10/shrec-shared/lassi/tmp/x"
    assert [p.pattern for p in deny_patterns() if p.search(cmd)] == []
