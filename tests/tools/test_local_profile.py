"""Tests for tools/local/host-profile.sh, the workstation's host profile (task P5.15).

Bible: Host Facts (Workstation, Workstation scratch), Agent Rule 7; AGENTS.md
Local Execution; plans/p5-ir.md P5.15 ("Host profile"); plans/local-plan.md
section 3.3, item 1, and section 7, decision 3 (b).

The contract these tests fix:

- The profile is sourced, never executed, in bash:
  LASSI_LOCAL_ROOT=<abs scratch root> LASSI_LOCAL_DRIVE=<drive mount, /mnt/c or /mnt/z>
  LASSI_LOCAL_MEMORY_MAX=<systemd size such as 20G> source tools/local/host-profile.sh
- Each of the three inputs has no default. A missing or bad one prints a
  message naming it to stderr and returns 2, sets nothing, and never exits the
  caller's shell (also under `set -u`).
- It refuses a root that is relative, is "/", or lies under /mnt (where WSL
  mounts the Windows drives), /tmp, /var/tmp, or /opt.
- It exports every variable tools/server/gate.py build_env sets, with the same
  sub-paths under the root (read here from gate.py with the ast module, so the
  two cannot drift), LASSI_JOBS=8, root/bin at the front of PATH, and HOME at
  the root; plus LASSI_HOST=workstation, LASSI_ALLOWED_ROOT, LASSI_STOP_GIB=145,
  LASSI_CAP_GIB=150, LASSI_DRIVE_FLOOR_GIB=100, LASSI_HOST_DRIVE, and
  LASSI_JOB_MEMORY_MAX. It creates the directories build_env creates.
- It defines lassi_nice, which runs "$@" under "ionice -c2 -n7 nice -n 10",
  ionice only when present, as gate.py's wrap does with nice 10.

The tests run the profile in the bash on PATH with a scratch root under the
test's temporary directory, named as that bash names it (on Windows the drive
form /c/..., which lies under no refused prefix). nice and ionice are stand-in
scripts that log their arguments; no test runs systemd-run, prlimit, wsl.exe,
or a real nice. No value here is a measurement.
"""

from __future__ import annotations

import ast
import os
import shutil
import subprocess
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PROFILE = REPO / "tools" / "local" / "host-profile.sh"
GATE = REPO / "tools" / "server" / "gate.py"
# The bash on PATH, resolved here: on Windows a bare "bash" given to subprocess can start the WSL launcher in
# System32 instead, which sees neither this environment nor Windows paths (tests/toolchains/test_tt_pins.py).
BASH = shutil.which("bash")
needs_bash = pytest.mark.skipif(BASH is None, reason="bash is not available")
INPUTS = ("LASSI_LOCAL_ROOT", "LASSI_LOCAL_DRIVE", "LASSI_LOCAL_MEMORY_MAX")
# The names plans/p5-ir.md P5.15 lists for build_env, PATH and PYTHONUNBUFFERED included; a check that the reader
# of gate.py below finds them, so a reader bug cannot make the drift test pass with nothing to compare.
PLAN_NAMES = {
    "LASSI_SCRATCH", "LASSI_RUNS_ROOT", "LASSI_TOOLCHAINS", "LASSI_JOBS", "TMPDIR", "TMP", "TEMP",
    "XDG_CACHE_HOME", "UV_CACHE_DIR", "PIP_CACHE_DIR", "HF_HOME", "TIKTOKEN_CACHE_DIR", "CCACHE_DIR",
    "CARGO_HOME", "RUSTUP_HOME", "DOTNET_CLI_HOME", "NUGET_PACKAGES", "npm_config_cache", "PATH",
    "PYTHONUNBUFFERED",
}
# build_env values the reader cannot evaluate to a path, each checked by its own rule here.
COMPUTED = {"LASSI_JOBS", "PATH"}
# A stand-in for nice or ionice: log the arguments, drop the two option words, and run the rest.
FAKE_TOOL = '#!/bin/sh\nprintf "%s\\n" "{name} $*" >> "$FAKE_LOG"\nshift 2\nexec "$@"\n'
SCRIPT = """\
__before_path=$PATH
source "$1"
__rc=$?
printf '__rc=%s\\n' "$__rc"
printf '__before_path=%s\\n' "$__before_path"
if declare -F lassi_nice >/dev/null; then printf '__nice=yes\\n'; else printf '__nice=no\\n'; fi
if [ -n "${LASSI_LOCAL_ROOT:-}" ] && [ -e "$LASSI_LOCAL_ROOT" ]; then printf '__root_exists=yes\\n'; fi
"""
SCRIPT_END = """\
env -0 > "$2"
printf '__alive=yes\\n'
"""


@dataclass
class Sourced:
    """What one bash session that sourced the profile reported."""

    rc: int | None
    alive: bool
    nice_defined: bool
    root_exists: bool
    before_path: str
    env: dict[str, str]
    stdout: str
    stderr: str


def bash_path(path: Path) -> str:
    """Return how the bash on PATH names `path`.

    On Windows that is the drive form /c/...; cygpath -u would give /tmp/... for the user's temporary directory,
    which the profile rightly refuses.
    """
    resolved = path.resolve()
    if os.name != "nt":
        return str(resolved)
    drive = resolved.drive
    return "/" + drive[0].lower() + resolved.as_posix()[len(drive):]


def require_profile() -> None:
    """Fail with the task's name when the profile does not exist yet."""
    if not PROFILE.is_file():
        pytest.fail("task P5.15 adds tools/local/host-profile.sh")


def base_env(home: Path) -> dict[str, str]:
    """Return this process's environment without any LASSI_ variable, with HOME at `home`."""
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith("LASSI_")}
    env["HOME"] = bash_path(home)
    return env


def source_profile(
    tmp_path: Path, inputs: Mapping[str, str | None], *, prelude: str = "", after: str = ""
) -> Sourced:
    """Source the profile in bash with `inputs` (None leaves a variable unset) and report the session."""
    require_profile()
    home = tmp_path / "home-before"
    home.mkdir(exist_ok=True)
    env = base_env(home)
    for name, value in inputs.items():
        if value is not None:
            env[name] = value
    dump = tmp_path / f"env-{uuid.uuid4().hex}.bin"
    script = prelude + SCRIPT + after + SCRIPT_END
    proc = subprocess.run(
        [BASH, "-c", script, "bash", bash_path(PROFILE), bash_path(dump)],
        env=env, cwd=tmp_path, capture_output=True, text=True, timeout=120,
    )
    marks = dict(line.split("=", 1) for line in proc.stdout.splitlines() if line.startswith("__") and "=" in line)
    dumped: dict[str, str] = {}
    if dump.is_file():
        for entry in dump.read_bytes().decode("utf-8", errors="replace").split("\0"):
            key, sep, value = entry.partition("=")
            if sep and key:
                dumped[key] = value
    return Sourced(
        rc=int(marks["__rc"]) if "__rc" in marks else None,
        alive=marks.get("__alive") == "yes",
        nice_defined=marks.get("__nice") == "yes",
        root_exists=marks.get("__root_exists") == "yes",
        before_path=marks.get("__before_path", ""),
        env=dumped,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


def good_inputs(root: Path, drive: str = "/mnt/c", memory: str = "20G") -> dict[str, str | None]:
    """Return the three inputs with a valid root, drive, and memory cap."""
    return {"LASSI_LOCAL_ROOT": bash_path(root), "LASSI_LOCAL_DRIVE": drive, "LASSI_LOCAL_MEMORY_MAX": memory}


def make_root(tmp_path: Path) -> Path:
    """Return a new scratch root under the test's temporary directory."""
    root = tmp_path / "scratch"
    root.mkdir()
    for prefix in ("/mnt/", "/tmp/", "/var/tmp/", "/opt/"):
        if (bash_path(root) + "/").startswith(prefix):
            pytest.skip(f"the test's temporary directory lies under {prefix}, which the profile refuses; set TMPDIR")
    return root


# Reading build_env from gate.py.


def gate_tree() -> ast.Module:
    """Return the parsed tools/server/gate.py."""
    return ast.parse(GATE.read_text(encoding="utf-8"))


def build_env_node(tree: ast.Module) -> ast.FunctionDef:
    """Return gate.py's build_env function."""
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "build_env":
            return node
    raise AssertionError("tools/server/gate.py has no build_env")


def build_env_dict(tree: ast.Module) -> dict[str, ast.expr]:
    """Return the keys and value expressions of the dict build_env passes to env.update."""
    for node in ast.walk(build_env_node(tree)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "update"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "env"
            and node.args
            and isinstance(node.args[0], ast.Dict)
        ):
            pairs = zip(node.args[0].keys, node.args[0].values, strict=True)
            return {key.value: value for key, value in pairs if isinstance(key, ast.Constant)}
    raise AssertionError("build_env passes no dict literal to env.update")


def build_env_dirs(tree: ast.Module) -> list[str]:
    """Return the sub-directories of the scratch root build_env creates (its `for sub in (...)` loop)."""
    for node in ast.walk(build_env_node(tree)):
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Tuple):
            return [item.value for item in node.iter.elts if isinstance(item, ast.Constant)]
    raise AssertionError("build_env creates no directories")


def module_names(tree: ast.Module) -> dict[str, ast.expr]:
    """Return gate.py's module-level assignments by name."""
    names: dict[str, ast.expr] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names[target.id] = node.value
    return names


def evaluate(expr: ast.expr, names: Mapping[str, ast.expr], root: str) -> str | None:
    """Evaluate a build_env value with SCRATCH at `root`: constants, names, and os.path.join; None otherwise."""
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return expr.value
    if isinstance(expr, ast.Name):
        if expr.id == "SCRATCH":
            return root
        return evaluate(names[expr.id], names, root) if expr.id in names else None
    if isinstance(expr, ast.Call) and ast.unparse(expr.func) == "os.path.join":
        parts = [evaluate(arg, names, root) for arg in expr.args]
        return None if None in parts else "/".join(part for part in parts if part is not None)
    return None


def expected_paths(root: str) -> dict[str, str | None]:
    """Return each build_env name with the value it takes under `root` (None for a computed value)."""
    tree = gate_tree()
    names = module_names(tree)
    return {key: evaluate(value, names, root) for key, value in build_env_dict(tree).items()}


def test_the_reader_of_gate_py_finds_build_env_names() -> None:
    """A check on this module's reader of gate.py, not on the profile: it finds the plan's names."""
    expected = expected_paths("/root")
    assert PLAN_NAMES <= set(expected), sorted(PLAN_NAMES - set(expected))
    assert {key for key, value in expected.items() if value is None} == COMPUTED
    assert expected["LASSI_RUNS_ROOT"] == "/root/lassi-runs"
    assert expected["TMPDIR"] == "/root/tmp"
    assert build_env_dirs(gate_tree()) == ["tmp", ".cache", "bin"]


# The profile's environment.


@needs_bash
def test_profile_exports_every_build_env_variable_with_gate_sub_paths(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    sourced = source_profile(tmp_path, good_inputs(root))
    assert sourced.rc == 0, sourced.stderr
    assert sourced.alive, sourced.stderr
    expected = expected_paths(bash_path(root))
    missing = sorted(name for name in expected if name not in sourced.env)
    assert not missing, f"the profile does not export {missing}"
    for name, value in expected.items():
        if value is not None:
            assert sourced.env[name] == value, name


@needs_bash
def test_profile_sets_jobs_path_and_unbuffered_python(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    sourced = source_profile(tmp_path, good_inputs(root))
    assert sourced.rc == 0, sourced.stderr
    assert sourced.env.get("LASSI_JOBS") == "8"
    assert sourced.env.get("PYTHONUNBUFFERED") == "1"
    assert sourced.before_path, "the session printed no PATH from before the profile"
    assert sourced.env.get("PATH") == bash_path(root) + "/bin:" + sourced.before_path


@needs_bash
def test_profile_sets_home_and_the_workstation_policy_values(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    sourced = source_profile(tmp_path, good_inputs(root, drive="/mnt/z", memory="24G"))
    assert sourced.rc == 0, sourced.stderr
    expected = {
        "HOME": bash_path(root),
        "LASSI_HOST": "workstation",
        "LASSI_ALLOWED_ROOT": bash_path(root),
        "LASSI_STOP_GIB": "145",
        "LASSI_CAP_GIB": "150",
        "LASSI_DRIVE_FLOOR_GIB": "100",
        "LASSI_HOST_DRIVE": "/mnt/z",
        "LASSI_JOB_MEMORY_MAX": "24G",
    }
    assert {name: sourced.env.get(name) for name in expected} == expected


@needs_bash
def test_profile_accepts_drive_c_and_reads_the_memory_cap(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    sourced = source_profile(tmp_path, good_inputs(root, drive="/mnt/c", memory="20G"))
    assert sourced.rc == 0, sourced.stderr
    assert sourced.env.get("LASSI_HOST_DRIVE") == "/mnt/c"
    assert sourced.env.get("LASSI_JOB_MEMORY_MAX") == "20G"


@needs_bash
def test_profile_creates_the_directories_build_env_creates(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    sourced = source_profile(tmp_path, good_inputs(root))
    assert sourced.rc == 0, sourced.stderr
    for sub in build_env_dirs(gate_tree()):
        assert (root / sub).is_dir(), f"the profile did not create {sub} under the root"


@needs_bash
def test_profile_sources_cleanly_under_set_u(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    sourced = source_profile(tmp_path, good_inputs(root), prelude="set -u\n")
    assert sourced.rc == 0, sourced.stderr
    assert sourced.alive, sourced.stderr
    assert sourced.env.get("LASSI_SCRATCH") == bash_path(root)


# Refusals.


def assert_refused(sourced: Sourced, home: Path, name: str) -> None:
    """Assert the profile returned 2 naming `name`, kept the shell running, and set nothing."""
    assert sourced.rc == 2, (sourced.rc, sourced.stderr)
    assert sourced.alive, f"the profile ended the caller's shell: {sourced.stderr}"
    assert name in sourced.stderr, f"the message does not name {name}: {sourced.stderr!r}"
    assert "LASSI_SCRATCH" not in sourced.env, "a refused profile still exported LASSI_SCRATCH"
    assert sourced.env.get("HOME") == bash_path(home), "a refused profile still moved HOME"


@needs_bash
@pytest.mark.parametrize("missing", INPUTS)
@pytest.mark.parametrize("prelude", ["", "set -u\n"], ids=["plain", "set-u"])
def test_profile_refuses_each_missing_input(tmp_path: Path, missing: str, prelude: str) -> None:
    root = make_root(tmp_path)
    inputs = good_inputs(root)
    inputs[missing] = None
    sourced = source_profile(tmp_path, inputs, prelude=prelude)
    assert_refused(sourced, tmp_path / "home-before", missing)


@needs_bash
@pytest.mark.parametrize("empty", INPUTS)
def test_profile_refuses_each_empty_input(tmp_path: Path, empty: str) -> None:
    root = make_root(tmp_path)
    inputs = good_inputs(root)
    inputs[empty] = ""
    sourced = source_profile(tmp_path, inputs)
    assert_refused(sourced, tmp_path / "home-before", empty)


UNIQUE = uuid.uuid4().hex[:12]
REFUSED_ROOTS = [
    "/",
    "/mnt",
    f"/mnt/c/lassi-refused-{UNIQUE}",
    f"/mnt/z/lassi-refused-{UNIQUE}",
    f"/mnt/wsl/lassi-refused-{UNIQUE}",
    "/tmp",
    f"/tmp/lassi-refused-{UNIQUE}",
    f"/var/tmp/lassi-refused-{UNIQUE}",
    f"/opt/lassi-refused-{UNIQUE}",
    f"lassi-refused-{UNIQUE}",
    f"./lassi-refused-{UNIQUE}",
]


@needs_bash
@pytest.mark.parametrize("root", REFUSED_ROOTS)
def test_profile_refuses_a_root_outside_the_distro_disk_policy(tmp_path: Path, root: str) -> None:
    make_root(tmp_path)
    inputs = good_inputs(tmp_path / "scratch")
    inputs["LASSI_LOCAL_ROOT"] = root
    sourced = source_profile(tmp_path, inputs)
    assert_refused(sourced, tmp_path / "home-before", "LASSI_LOCAL_ROOT")
    if UNIQUE in root:
        assert not sourced.root_exists, f"the profile created the refused root {root}"
        assert not (tmp_path / root.removeprefix("./")).exists(), "the profile created a relative root"


@needs_bash
@pytest.mark.parametrize("drive", ["c", "C:", "/c", "/mnt", "/mnt/c/Users", "/tmp"])
def test_profile_refuses_a_drive_that_is_not_a_drive_mount(tmp_path: Path, drive: str) -> None:
    root = make_root(tmp_path)
    sourced = source_profile(tmp_path, good_inputs(root, drive=drive))
    assert_refused(sourced, tmp_path / "home-before", "LASSI_LOCAL_DRIVE")


@needs_bash
@pytest.mark.parametrize("memory", ["lots", "infinity", "-1G", "20 G"])
def test_profile_refuses_a_memory_cap_that_is_not_a_bounded_size(tmp_path: Path, memory: str) -> None:
    root = make_root(tmp_path)
    sourced = source_profile(tmp_path, good_inputs(root, memory=memory))
    assert_refused(sourced, tmp_path / "home-before", "LASSI_LOCAL_MEMORY_MAX")


# lassi_nice.


def fake_tools(root: Path, names: tuple[str, ...]) -> Path:
    """Put stand-in nice (and ionice) scripts in root/bin, which the profile puts first on PATH; return the log."""
    bin_dir = root / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        text = FAKE_TOOL.replace("{name}", name)
        path = bin_dir / name
        path.write_bytes(text.encode("ascii"))
        path.chmod(0o755)
    return root / "fake.log"


@needs_bash
def test_profile_defines_lassi_nice(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    sourced = source_profile(tmp_path, good_inputs(root))
    assert sourced.rc == 0, sourced.stderr
    assert sourced.nice_defined, "the profile defines no lassi_nice function"


@needs_bash
def test_lassi_nice_runs_the_command_under_ionice_and_nice(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    log = fake_tools(root, ("ionice", "nice"))
    after = f"export FAKE_LOG='{bash_path(log)}'\nlassi_nice printf '%s\\n' ran-under-nice\n"
    after += "lassi_nice sh -c 'exit 7'\nprintf '__status=%s\\n' \"$?\"\n"
    sourced = source_profile(tmp_path, good_inputs(root), after=after)
    assert sourced.rc == 0, sourced.stderr
    assert "ran-under-nice" in sourced.stdout.splitlines()
    assert "__status=7" in sourced.stdout.splitlines(), "lassi_nice does not return the command's status"
    lines = log.read_text(encoding="ascii").splitlines()
    assert lines[:2] == [
        "ionice -c2 -n7 nice -n 10 printf %s\\n ran-under-nice",
        "nice -n 10 printf %s\\n ran-under-nice",
    ]


@needs_bash
def test_lassi_nice_skips_ionice_when_it_is_absent(tmp_path: Path) -> None:
    probe = subprocess.run([BASH, "-c", "command -v ionice"], capture_output=True, text=True, timeout=60)
    if probe.returncode == 0:
        pytest.skip("this host has an ionice on PATH; the case without one runs where there is none")
    root = make_root(tmp_path)
    log = fake_tools(root, ("nice",))
    after = f"export FAKE_LOG='{bash_path(log)}'\nlassi_nice printf '%s\\n' ran-under-nice\n"
    sourced = source_profile(tmp_path, good_inputs(root), after=after)
    assert sourced.rc == 0, sourced.stderr
    assert "ran-under-nice" in sourced.stdout.splitlines(), sourced.stderr
    assert log.read_text(encoding="ascii").splitlines() == ["nice -n 10 printf %s\\n ran-under-nice"]


def test_profile_is_plain_ascii_with_lf_line_endings() -> None:
    require_profile()
    data = PROFILE.read_bytes()
    data.decode("ascii")
    assert b"\r" not in data, "the profile has CR line endings, which bash in the distro reads as part of a word"
