"""Tests for tools/local/provenance.py, the local provenance writer (task P5.15, the Windows half).

The writer stands in for the provenance.json that `rx pull` copies from the build host's gate
(tools/rx.py, pull; tools/server/gate.py, base_meta, host_info, and read_pins) when a run happens in
the LASSI distro on the workstation (plans/p5-ir.md, P5.15, the local provenance writer; AGENTS.md,
Results, local runs; Agent Rules 1 and 2). It runs under the distro's system python3.10 with the
standard library only. These tests run on Windows with fakes only: a throwaway git repository under
tmp_path stands for the clone, fake os-release, cpuinfo, meminfo, and kernel osrelease files stand
for /etc and /proc, and fake functions stand for the util-linux version commands and the host name.
No test runs systemd-run, prlimit, wsl.exe, apt, or curl, and every host value in the fakes is a
PLACEHOLDER, not a measurement.

The interface these tests fix:

- `HostSources(os_release, cpuinfo, meminfo, kernel_release, version_output, hostname)` says where
  the writer reads the host. The paths default to /etc/os-release, /proc/cpuinfo, /proc/meminfo,
  and /proc/sys/kernel/osrelease. `version_output(argv)` returns the stdout of a version command,
  `findmnt --version` or else `mount --version`, and raises OSError when the command cannot run.
  `hostname()` returns the host's name. git runs on the clone itself and is not routed through it.
- `main(argv, *, environ=None, sources=None) -> int` reads LASSI_HOST, LASSI_SCRATCH, and
  WSL_DISTRO_NAME from `environ` (default os.environ) and the host from `sources` (default
  HostSources()). It returns 0 after writing --out. On a refusal it prints a clear message to
  stderr, writes nothing, and returns 2; argparse's usage errors exit 2 as well.

The record keeps the gate's key names where the meaning is the same (cmd, cwd, commit, dirty,
branch, start, end, rc, host, pins; in host: hostname, os, kernel, nproc, mem_gb, python) and adds
kind "local"; in host, kind "workstation", distro, os_version, util_linux, and cpu_model; device;
and date, the writer's own run date. The job record of tools/local/longjob.py is read for its command
(a list of words, joined with shell quoting into cmd), exit_status (rc), start, end, cwd, commit, dirty,
and host, the names tests/tools/test_local_longjob.py fixes for that record. A job whose commit is not
a 40-character lowercase hex string or whose dirty is not true or false is refused: its tree state was
not recorded, so it runs again from inside the clone. A job at another commit than the clone's HEAD,
or from another host, or run outside the scratch root, is refused, and a job that ran dirty makes the
record dirty. The alpha01 check skips only the verbatim pin texts, whose
committed comments name alpha01 as where a pin value was measured.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import os
import platform
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
WRITER = REPO / "tools" / "local" / "provenance.py"

IDENTITY = {
    "GIT_AUTHOR_NAME": "Provenance Test",
    "GIT_AUTHOR_EMAIL": "provenance-test@example.invalid",
    "GIT_COMMITTER_NAME": "Provenance Test",
    "GIT_COMMITTER_EMAIL": "provenance-test@example.invalid",
}

# PLACEHOLDER host facts for the fakes; none of them is a measurement of any host.
PLACEHOLDER_HOSTNAME = "PLACEHOLDER-WORKSTATION"
PLACEHOLDER_DISTRO = "PLACEHOLDER-LASSI-DISTRO"
PLACEHOLDER_PRETTY_NAME = "PLACEHOLDER Linux 0.0 LTS"
PLACEHOLDER_VERSION_ID = "0.0"
PLACEHOLDER_KERNEL = "0.0.0-PLACEHOLDER-microsoft-standard-WSL2"
PLACEHOLDER_CPU_MODEL = "PLACEHOLDER CPU Model 9000 @ 0.00GHz"
PLACEHOLDER_MEM_KB = 12345678
PLACEHOLDER_FINDMNT_VERSION = "2.99.1"
PLACEHOLDER_MOUNT_VERSION = "2.99.2"
PLACEHOLDER_START = "2026-01-02T03:04:05+00:00"
PLACEHOLDER_END = "2026-01-02T03:09:05+00:00"

OS_RELEASE = (
    f'PRETTY_NAME="{PLACEHOLDER_PRETTY_NAME}"\n'
    'NAME="PLACEHOLDER Linux"\n'
    # VERSION comes before VERSION_ID, so a bare prefix match on VERSION reads the wrong line.
    'VERSION="0.0 LTS (Placeholder)"\n'
    f'VERSION_ID="{PLACEHOLDER_VERSION_ID}"\n'
    "ID=placeholder\n"
)
CPUINFO = (
    "processor\t: 0\n"
    "vendor_id\t: PlaceholderVendor\n"
    "cpu family\t: 0\n"
    # A bare "model" line comes before "model name", as in a real cpuinfo.
    "model\t\t: 0\n"
    f"model name\t: {PLACEHOLDER_CPU_MODEL}\n"
    "\n"
    "processor\t: 1\n"
    "vendor_id\t: PlaceholderVendor\n"
    "cpu family\t: 0\n"
    "model\t\t: 0\n"
    f"model name\t: {PLACEHOLDER_CPU_MODEL}\n"
)
MEMINFO = f"MemTotal:       {PLACEHOLDER_MEM_KB} kB\nMemFree:          100000 kB\nMemAvailable:     200000 kB\n"
FINDMNT_OUTPUT = f"findmnt from util-linux {PLACEHOLDER_FINDMNT_VERSION}\n"
MOUNT_OUTPUT = (
    f"mount from util-linux {PLACEHOLDER_MOUNT_VERSION} "
    f"(libmount {PLACEHOLDER_MOUNT_VERSION}: selinux, smack, btrfs, namespaces, assert, debug)\n"
)

FAKE_PINS = {
    "gcc.pin": "# PLACEHOLDER gcc pin for the provenance tests.\nNAME=gcc\nVERSION=0.0.0\n",
    "ttsim.pin": "# PLACEHOLDER ttsim pin.\nNAME=ttsim\nVERSION=v9.8.7\nPREFIX_NAME=ttsim@v9.8.7\n",
    # VERSION differs from COMMIT's first 8 characters on purpose: the device string reads COMMIT.
    "tt-metal.pin": (
        "# PLACEHOLDER tt-metal pin.\nNAME=tt-metal\nCOMMIT=abcdef1234567890abcdef1234567890abcdef12\n"
        "VERSION=placeholder\nPREFIX_NAME=tt-metal@placeholder\n"
    ),
    # Longer than the 4000 characters the gate's read_pins keeps.
    "long.pin": "# PLACEHOLDER long pin.\n" + "".join(f"LINE_{i:04d}=placeholder-value\n" for i in range(300)),
}
NOT_PINS = {"tt-metal.sh": "#!/usr/bin/env bash\n", "tt-metal-cpm-sources.txt": "placeholder\n"}
FAKE_TTSIM_DEVICE = "ttsim v9.8.7 on tt-metal abcdef12"
# The bible names today's simulator device so (Host Facts, Workstation evidence).
TODAY_TTSIM_DEVICE = "ttsim v1.3.4 on tt-metal 5280a9cf"

TOP_KEYS = ("kind", "cmd", "cwd", "commit", "dirty", "branch", "start", "end", "rc", "host", "pins", "device", "date")
OPTIONS = ("--out", "--device", "--job", "--cmd", "--rc", "--start", "--end", "--repo")


def explicit(cmd: str = "make check", rc: str = "3") -> list[str]:
    """The explicit route's arguments: command, exit status, start, and end."""
    return ["--cmd", cmd, "--rc", rc, "--start", PLACEHOLDER_START, "--end", PLACEHOLDER_END]


EXPLICIT = explicit()


# ---------------------------------------------------------------------------
# Helpers


@lru_cache(maxsize=1)
def _load_writer() -> ModuleType:
    spec = importlib.util.spec_from_file_location("lassi_local_provenance_under_test", WRITER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before it runs, as an import would be: dataclasses look their module up in sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def writer() -> ModuleType:
    """Import tools/local/provenance.py by path (a standalone script); fail clearly while it is missing."""
    if not WRITER.is_file():
        pytest.fail(f"{WRITER.relative_to(REPO).as_posix()} does not exist yet")
    return _load_writer()


def git(repo: Path, *args: str) -> str:
    """Run git in the throwaway `repo` with the test identity; return its stdout, stripped."""
    done = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, env={**os.environ, **IDENTITY}, timeout=60, check=True,
    )
    return done.stdout.strip()


def make_clone(parent: Path, pins: Mapping[str, str]) -> Path:
    """A throwaway clone under `parent` holding `pins` and two non-pin files in toolchains/, with one commit."""
    clone = parent / "lassi"
    (clone / "toolchains").mkdir(parents=True)
    (clone / "sub").mkdir()
    for name, text in {**pins, **NOT_PINS}.items():
        (clone / "toolchains" / name).write_bytes(text.encode("ascii"))
    (clone / "README.md").write_bytes(b"A throwaway clone for the provenance tests.\n")
    (clone / "sub" / "keep.txt").write_bytes(b"placeholder\n")
    git(clone, "init", "--quiet", "-b", "main")
    git(clone, "config", "core.autocrlf", "false")
    git(clone, "add", "-A")
    git(clone, "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "P0.99: a throwaway base")
    return clone


def fake_version_output(outputs: Mapping[str, str]) -> Callable[[Sequence[str]], str]:
    """A version command stand-in: the output named by the command's base name, else FileNotFoundError."""
    def version_output(argv: Sequence[str]) -> str:
        name = os.path.basename(str(argv[0]))
        if name not in outputs:
            raise FileNotFoundError(f"PLACEHOLDER: no {name} on the fake host")
        return outputs[name]
    return version_output


def fake_sources(module: ModuleType, root: Path, hostname: str, versions: Mapping[str, str]) -> Any:
    """HostSources over fake /etc and /proc files under `root`, with fake commands."""
    files = {
        ("etc", "os-release"): OS_RELEASE,
        ("proc", "cpuinfo"): CPUINFO,
        ("proc", "meminfo"): MEMINFO,
        ("proc", "sys", "kernel", "osrelease"): PLACEHOLDER_KERNEL + "\n",
    }
    for parts, text in files.items():
        path = root.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("ascii"))
    return module.HostSources(
        os_release=root / "etc" / "os-release",
        cpuinfo=root / "proc" / "cpuinfo",
        meminfo=root / "proc" / "meminfo",
        kernel_release=root / "proc" / "sys" / "kernel" / "osrelease",
        version_output=fake_version_output(versions),
        hostname=lambda: hostname,
    )


def base_environ(scratch: Path) -> dict[str, str]:
    """The caller's environment without its LASSI_* and WSL_DISTRO_NAME, plus the host profile's values."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("LASSI_") and k != "WSL_DISTRO_NAME"}
    env.update({"LASSI_HOST": "workstation", "LASSI_SCRATCH": str(scratch), "WSL_DISTRO_NAME": PLACEHOLDER_DISTRO})
    return env


@dataclass
class Site:
    """One fake workstation: the writer, a scratch root, a clone, an output path, the host, and the environment."""

    module: ModuleType
    capsys: pytest.CaptureFixture[str]
    scratch: Path
    clone: Path
    out: Path
    sources: Any
    environ: dict[str, str]

    def argv(self, *route: str, device: str = "cpu", repo: bool = True, out: Path | None = None) -> list[str]:
        """The writer's arguments for `route`, with --out, --device, and (unless repo is False) --repo."""
        args = ["--out", str(out or self.out), "--device", device, *route]
        return args + (["--repo", str(self.clone)] if repo else [])

    def run(self, argv: list[str], environ: Mapping[str, str] | None = None) -> tuple[int, str]:
        """Call main; return its exit status (argparse's SystemExit included) and its stderr."""
        self.capsys.readouterr()
        try:
            rc = self.module.main(argv, environ=dict(self.environ if environ is None else environ),
                                  sources=self.sources)
        except SystemExit as stop:
            rc = stop.code if isinstance(stop.code, int) else (0 if stop.code is None else 1)
        return rc, self.capsys.readouterr().err

    def record(self) -> dict[str, Any]:
        """The written provenance.json, parsed."""
        return json.loads(self.out.read_text(encoding="ascii"))

    def job(self, **fields: Any) -> Path:
        """A job record in the shape tools/local/longjob.py writes (tests/tools/test_local_longjob.py, the
        record's fields), with PLACEHOLDER values, under the runs root."""
        record = {
            "id": "PLACEHOLDER-job", "name": "PLACEHOLDER-job", "command": ["bash", "toolchains/ttsim.sh"],
            "cwd": str(self.clone), "commit": git(self.clone, "rev-parse", "HEAD"), "dirty": False,
            "start": PLACEHOLDER_START, "end": PLACEHOLDER_END, "exit_status": 7,
            "scratch_kib_before": None, "scratch_kib_after": None,
            "drive_free_kib_before": None, "drive_free_kib_after": None,
            "memory_max_requested": "PLACEHOLDER", "memory_max_read": None, "memory_peak": None,
            "memory_peak_reason": "PLACEHOLDER: no scope in the fakes",
            "host": "workstation", "hostname": PLACEHOLDER_HOSTNAME,
        }
        record.update(fields)
        path = self.scratch / "lassi-runs" / "jobs" / "PLACEHOLDER-job" / "job.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record), encoding="ascii")
        return path


def make_site(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    *,
    clone_parent: Path | None = None,
    pins: Mapping[str, str] = FAKE_PINS,
    hostname: str = PLACEHOLDER_HOSTNAME,
    versions: Mapping[str, str] | None = None,
) -> Site:
    """A Site whose clone sits under the scratch root unless `clone_parent` says otherwise."""
    module = writer()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    clone = make_clone(clone_parent or scratch, pins)
    out = scratch / "out" / "provenance.json"
    out.parent.mkdir(parents=True)
    sources = fake_sources(module, tmp_path / "fakehost", hostname,
                           {"findmnt": FINDMNT_OUTPUT} if versions is None else versions)
    return Site(module, capsys, scratch, clone, out, sources, base_environ(scratch))


def today_pins() -> dict[str, str]:
    """This repository's ttsim and tt-metal pins, cut to the lines the device string reads."""
    keep = {"ttsim.pin": ("NAME=", "VERSION="), "tt-metal.pin": ("NAME=", "COMMIT=", "VERSION=")}
    pins = {}
    for name, prefixes in keep.items():
        text = (REPO / "toolchains" / name).read_text(encoding="ascii")
        pins[name] = "".join(line + "\n" for line in text.splitlines() if line.startswith(prefixes))
    return pins


# ---------------------------------------------------------------------------
# The script itself


def test_the_writer_is_python_310_standard_library_with_linux_modules_imported_lazily() -> None:
    writer()  # it loads on Windows, so nothing Linux-only runs at import
    tree = ast.parse(WRITER.read_text(encoding="ascii"), feature_version=(3, 10))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])
    assert sorted(name for name in imported if name not in sys.stdlib_module_names) == []
    top: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            top.add(node.module.split(".")[0])
    assert top.isdisjoint({"fcntl", "resource"}), sorted(top)


def test_the_script_runs_as_a_command_and_lists_its_options() -> None:
    writer()
    done = subprocess.run([sys.executable, str(WRITER), "--help"], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    missing = [option for option in OPTIONS if option not in done.stdout]
    assert missing == [], done.stdout


def test_host_sources_default_to_the_real_host_paths() -> None:
    sources = writer().HostSources()
    assert Path(sources.os_release).as_posix() == "/etc/os-release"
    assert Path(sources.cpuinfo).as_posix() == "/proc/cpuinfo"
    assert Path(sources.meminfo).as_posix() == "/proc/meminfo"
    assert Path(sources.kernel_release).as_posix() == "/proc/sys/kernel/osrelease"
    assert callable(sources.version_output) and callable(sources.hostname)


def test_a_docstring_notes_that_the_gate_truncates_pins_at_4000_characters_and_this_writer_does_not() -> None:
    module = writer()
    docs = [module.__doc__ or ""]
    docs += [inspect.getdoc(obj) or "" for _, obj in inspect.getmembers(module, inspect.isfunction)]
    assert any("4000" in doc and "truncat" in doc.lower() for doc in docs)


# ---------------------------------------------------------------------------
# The clone: commit, branch, clean or dirty


def test_a_clean_clone_records_its_commit_branch_and_clean_state(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    site = make_site(tmp_path, capsys)
    rc, err = site.run(site.argv(*EXPLICIT))
    assert rc == 0, err
    record = site.record()
    assert record["kind"] == "local"
    assert record["commit"] == git(site.clone, "rev-parse", "HEAD")
    assert record["branch"] == "main"
    assert record["dirty"] is False
    assert Path(record["cwd"]).resolve() == site.clone.resolve()


@pytest.mark.parametrize("change", ["modified-tracked-file", "untracked-file"])
def test_a_changed_clone_reads_dirty(tmp_path: Path, capsys: pytest.CaptureFixture[str], change: str) -> None:
    site = make_site(tmp_path, capsys)
    if change == "modified-tracked-file":
        (site.clone / "README.md").write_bytes(b"changed after the commit\n")
    else:
        (site.clone / "sub" / "untracked.txt").write_bytes(b"never added\n")
    assert git(site.clone, "status", "--porcelain"), "the fake change is not visible to git"
    rc, err = site.run(site.argv(*EXPLICIT))
    assert rc == 0, err
    assert site.record()["dirty"] is True


def test_the_clone_defaults_to_the_git_top_level_of_the_cwd(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    site = make_site(tmp_path, capsys)
    monkeypatch.chdir(site.clone / "sub")
    rc, err = site.run(site.argv(*EXPLICIT, repo=False))
    assert rc == 0, err
    record = site.record()
    assert record["commit"] == git(site.clone, "rev-parse", "HEAD")
    assert record["pins"] == FAKE_PINS


# ---------------------------------------------------------------------------
# The host


def test_the_host_block_names_the_workstation_and_its_facts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    site = make_site(tmp_path, capsys)
    rc, err = site.run(site.argv(*EXPLICIT))
    assert rc == 0, err
    host = site.record()["host"]
    assert host["kind"] == "workstation"
    assert host["hostname"] == PLACEHOLDER_HOSTNAME
    assert host["distro"] == PLACEHOLDER_DISTRO
    assert host["os"] == PLACEHOLDER_PRETTY_NAME
    assert host["os_version"] == PLACEHOLDER_VERSION_ID
    assert host["kernel"] == PLACEHOLDER_KERNEL
    assert host["util_linux"] == PLACEHOLDER_FINDMNT_VERSION
    assert host["nproc"] == os.cpu_count()
    assert host["cpu_model"] == PLACEHOLDER_CPU_MODEL
    assert host["mem_gb"] == round(PLACEHOLDER_MEM_KB / 1e6, 1)  # gate.py host_info's unit and rounding
    assert host["python"] == platform.python_version()


def test_util_linux_falls_back_to_mount_when_findmnt_cannot_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    site = make_site(tmp_path, capsys, versions={"mount": MOUNT_OUTPUT})
    rc, err = site.run(site.argv(*EXPLICIT))
    assert rc == 0, err
    assert site.record()["host"]["util_linux"] == PLACEHOLDER_MOUNT_VERSION


# ---------------------------------------------------------------------------
# Pins and the device


def test_pins_hold_the_full_text_of_every_pin_file_keyed_by_name(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert len(FAKE_PINS["long.pin"]) > 4000
    site = make_site(tmp_path, capsys)
    rc, err = site.run(site.argv(*EXPLICIT))
    assert rc == 0, err
    assert site.record()["pins"] == FAKE_PINS


def test_the_cpu_device_names_the_host_cpu_model(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    site = make_site(tmp_path, capsys)
    rc, err = site.run(site.argv(*EXPLICIT, device="cpu"))
    assert rc == 0, err
    device = site.record()["device"]
    assert device["kind"] == "cpu"
    assert device["model"] == PLACEHOLDER_CPU_MODEL
    assert device["simulator"] is False


def test_the_ttsim_device_is_a_simulator_named_from_the_clone_pins(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    site = make_site(tmp_path, capsys)
    rc, err = site.run(site.argv(*EXPLICIT, device="ttsim"))
    assert rc == 0, err
    device = site.record()["device"]
    assert device["kind"] == "ttsim"
    assert device["name"] == FAKE_TTSIM_DEVICE
    assert device["simulator"] is True
    note = device["note"].lower()
    assert "simulator" in note and "host cpu" in note and "never silicon" in note, device["note"]


def test_the_ttsim_device_reads_todays_pins_as_v1_3_4_on_tt_metal_5280a9cf(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    site = make_site(tmp_path, capsys, pins=today_pins())
    rc, err = site.run(site.argv(*EXPLICIT, device="ttsim"))
    assert rc == 0, err
    assert site.record()["device"]["name"] == TODAY_TTSIM_DEVICE


def test_a_device_other_than_cpu_or_ttsim_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    site = make_site(tmp_path, capsys)
    rc, _ = site.run(site.argv(*EXPLICIT, device="gpu"))
    assert rc == 2
    assert not site.out.exists()


# ---------------------------------------------------------------------------
# The two routes: a job record, or the command, status, and dates given explicitly


def test_the_explicit_route_records_the_command_exit_status_start_and_end(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    site = make_site(tmp_path, capsys)
    rc, err = site.run(site.argv(*explicit(cmd="make check", rc="3")))
    assert rc == 0, err
    record = site.record()
    assert record["cmd"] == "make check"
    assert record["rc"] == 3
    assert record["start"] == PLACEHOLDER_START
    assert record["end"] == PLACEHOLDER_END


def test_the_job_route_takes_the_command_exit_status_start_and_end_from_the_job_record(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    site = make_site(tmp_path, capsys)
    job = site.job(command=["bash", "toolchains/ttsim.sh"], exit_status=7)
    rc, err = site.run(site.argv("--job", str(job)))
    assert rc == 0, err
    record = site.record()
    assert record["cmd"] == "bash toolchains/ttsim.sh"
    assert record["rc"] == 7
    assert record["start"] == PLACEHOLDER_START
    assert record["end"] == PLACEHOLDER_END
    assert record["commit"] == git(site.clone, "rev-parse", "HEAD")


@pytest.mark.parametrize("route", ["no-rc", "job-and-cmd", "neither"])
def test_the_routes_are_exclusive_and_the_explicit_one_needs_all_four_values(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], route: str
) -> None:
    site = make_site(tmp_path, capsys)
    args = {
        "no-rc": ["--cmd", "make check", "--start", PLACEHOLDER_START, "--end", PLACEHOLDER_END],
        "job-and-cmd": ["--job", str(site.job()), *EXPLICIT],
        "neither": [],
    }[route]
    rc, _ = site.run(site.argv(*args))
    assert rc == 2
    assert not site.out.exists()


def test_a_job_that_ran_dirty_makes_the_record_dirty(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    site = make_site(tmp_path, capsys)
    rc, err = site.run(site.argv("--job", str(site.job(dirty=True))))
    assert rc == 0, err
    assert site.record()["dirty"] is True


# What the refusal of a job whose tree state was never read says. longjob.py's git_state writes commit and dirty
# null when the job's cwd is not a git checkout, and dirty null when git status fails.
NOT_RECORDED = ("tree state was not recorded", "inside the clone")

JOB_FAULTS: dict[str, tuple[Callable[[Site], dict[str, Any]], tuple[str, ...]]] = {
    "another-commit": (lambda site: {"commit": "0" * 40}, ("commit",)),
    "another-host": (lambda site: {"host": "build-host"}, ("host",)),
    "a-cwd-outside-the-scratch-root": (
        lambda site: {"cwd": str(site.scratch.parent / "elsewhere")}, ("LASSI_SCRATCH",)
    ),
    "no-command": (lambda site: {"command": []}, ("command",)),
    "no-exit-status": (lambda site: {"exit_status": None}, ("exit_status",)),
    "a-start-without-an-offset": (lambda site: {"start": "2026-01-02T03:04:05"}, ("start",)),
    "a-null-commit": (lambda site: {"commit": None}, ("commit", *NOT_RECORDED)),
    "a-cwd-outside-a-checkout": (lambda site: {"commit": None, "dirty": None}, ("commit", *NOT_RECORDED)),
    "a-short-commit": (lambda site: {"commit": "abcdef12"}, ("commit", *NOT_RECORDED)),
    "an-uppercase-commit": (
        lambda site: {"commit": git(site.clone, "rev-parse", "HEAD").upper()}, ("commit", *NOT_RECORDED)
    ),
    "a-null-dirty": (lambda site: {"dirty": None}, ("dirty", *NOT_RECORDED)),
    "a-string-dirty": (lambda site: {"dirty": "false"}, ("dirty", *NOT_RECORDED)),
}


@pytest.mark.parametrize("fault", sorted(JOB_FAULTS))
def test_a_job_record_that_does_not_belong_to_the_clone_and_host_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], fault: str
) -> None:
    site = make_site(tmp_path, capsys)
    fields, words = JOB_FAULTS[fault]
    (site.scratch.parent / "elsewhere").mkdir()
    rc, err = site.run(site.argv("--job", str(site.job(**fields(site)))))
    assert rc == 2
    assert [word for word in words if word not in err] == [], err
    assert not site.out.exists()


def test_a_job_record_outside_the_scratch_root_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    site = make_site(tmp_path, capsys)
    inside = site.job()
    outside = tmp_path / "elsewhere" / "job.json"
    outside.parent.mkdir()
    outside.write_bytes(inside.read_bytes())
    rc, err = site.run(site.argv("--job", str(outside)))
    assert rc == 2
    assert "--job" in err and "LASSI_SCRATCH" in err, err
    assert not site.out.exists()


# ---------------------------------------------------------------------------
# Refusals


@pytest.mark.parametrize("value", [None, "", "build-host"], ids=["unset", "empty", "another-host"])
def test_a_host_other_than_the_workstation_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], value: str | None
) -> None:
    site = make_site(tmp_path, capsys)
    environ = {k: v for k, v in site.environ.items() if k != "LASSI_HOST"}
    if value is not None:
        environ["LASSI_HOST"] = value
    rc, err = site.run(site.argv(*EXPLICIT), environ=environ)
    assert rc == 2
    assert "LASSI_HOST" in err and "workstation" in err, err
    assert not site.out.exists()


@pytest.mark.parametrize("value", [None, ""], ids=["unset", "empty"])
def test_an_unset_scratch_root_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], value: str | None
) -> None:
    site = make_site(tmp_path, capsys)
    environ = {k: v for k, v in site.environ.items() if k != "LASSI_SCRATCH"}
    if value is not None:
        environ["LASSI_SCRATCH"] = value
    rc, err = site.run(site.argv(*EXPLICIT), environ=environ)
    assert rc == 2
    assert "LASSI_SCRATCH" in err, err
    assert not site.out.exists()


OUTSIDE_OUTS: dict[str, Callable[[Path], Path]] = {
    "a-sibling-directory": lambda scratch: scratch.parent / "elsewhere" / "provenance.json",
    "a-name-sharing-the-prefix": lambda scratch: scratch.parent / (scratch.name + "-other") / "provenance.json",
    "a-dot-dot-escape": lambda scratch: scratch / ".." / "elsewhere" / "provenance.json",
}


@pytest.mark.parametrize("where", sorted(OUTSIDE_OUTS))
def test_an_output_outside_the_scratch_root_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], where: str
) -> None:
    site = make_site(tmp_path, capsys)
    out = OUTSIDE_OUTS[where](site.scratch)
    Path(os.path.normpath(out.parent)).mkdir(parents=True, exist_ok=True)
    rc, err = site.run(site.argv(*EXPLICIT, out=out))
    assert rc == 2
    assert "--out" in err and "LASSI_SCRATCH" in err, err
    assert not Path(os.path.normpath(out)).exists()


def test_an_output_through_a_link_out_of_the_scratch_root_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    site = make_site(tmp_path, capsys)
    target = tmp_path / "elsewhere"
    target.mkdir()
    link = site.scratch / "link"
    try:
        os.symlink(target, link, target_is_directory=True)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"this host cannot make a directory symbolic link: {error}")
    rc, err = site.run(site.argv(*EXPLICIT, out=link / "provenance.json"))
    assert rc == 2
    assert "LASSI_SCRATCH" in err, err
    assert not (target / "provenance.json").exists()


def test_a_clone_outside_the_scratch_root_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    site = make_site(tmp_path, capsys, clone_parent=elsewhere)
    rc, err = site.run(site.argv(*EXPLICIT))
    assert rc == 2
    assert "LASSI_SCRATCH" in err and ("clone" in err.lower() or "--repo" in err), err
    assert not site.out.exists()


@pytest.mark.parametrize("where", ["cmd", "hostname", "distro", "job"])
def test_a_record_that_would_name_alpha01_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], where: str
) -> None:
    site = make_site(tmp_path, capsys, hostname="alpha01" if where == "hostname" else PLACEHOLDER_HOSTNAME)
    environ = dict(site.environ)
    route = EXPLICIT
    if where == "cmd":
        route = explicit(cmd="echo alpha01")
    elif where == "distro":
        environ["WSL_DISTRO_NAME"] = "alpha01-mirror"
    elif where == "job":
        route = ["--job", str(site.job(command=["echo", "copied from alpha01"]))]
    rc, err = site.run(site.argv(*route), environ=environ)
    assert rc == 2
    assert "alpha01" in err, err
    assert not site.out.exists()


def test_pin_comments_that_name_alpha01_are_kept_verbatim_and_not_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Committed pins name alpha01 in comments as where a value was measured (cuda.pin, gcc.pin, hipcc.pin,
    # tt-metal.pin); the full text is kept, so the alpha01 check skips the verbatim pin texts only.
    pins = {**FAKE_PINS, "gcc.pin": "# PLACEHOLDER: a value read on alpha01, as committed pins say.\nNAME=gcc\n"}
    site = make_site(tmp_path, capsys, pins=pins)
    rc, err = site.run(site.argv(*EXPLICIT))
    assert rc == 0, err
    assert site.record()["pins"] == pins


# ---------------------------------------------------------------------------
# The written file


def test_the_written_record_is_ascii_json_with_every_key_and_never_names_alpha01(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    site = make_site(tmp_path, capsys)
    rc, err = site.run(site.argv(*EXPLICIT, device="ttsim"))
    assert rc == 0, err
    text = site.out.read_bytes().decode("ascii")
    record = json.loads(text)
    assert isinstance(record, dict)
    assert [key for key in TOP_KEYS if key not in record] == []
    assert "alpha01" not in text


def test_the_record_carries_the_writers_own_run_date(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    site = make_site(tmp_path, capsys)
    before = datetime.now(timezone.utc)
    rc, err = site.run(site.argv(*EXPLICIT))
    after = datetime.now(timezone.utc)
    assert rc == 0, err
    date = datetime.fromisoformat(site.record()["date"])
    assert date.tzinfo is not None, "the run date carries its UTC offset, as the gate's dates do"
    assert before - timedelta(seconds=2) <= date <= after + timedelta(seconds=2)
