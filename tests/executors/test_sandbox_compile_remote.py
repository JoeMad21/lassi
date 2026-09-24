"""Remote tests for sandboxed compiles on the build host (P0.20; bible Sandbox, Toolchain Pins; Agent Rules 6, 7, 10).

These build real sandboxes, so they are marked `remote` and skip unless the
host can run them (the same needs as tests/executors/test_sandbox_remote.py:
Linux, the sandbox tools on PATH, a reachable user systemd manager, a
python3 in SANDBOX_PATH, and $LASSI_SCRATCH, $LASSI_RUNS_ROOT, and a TMPDIR
inside $LASSI_SCRATCH). The tests that compile with the real pinned nvcc and
nvc++ are also marked `slow` and need them under $LASSI_TOOLCHAINS. With
LASSI_REQUIRE_SANDBOX=1 a host that cannot run a test fails it instead of
skipping it, so a silent skip never passes for evidence. Run them with
`rx run -- 'LASSI_REQUIRE_SANDBOX=1 uv run pytest -q -m remote tests/executors/test_sandbox_compile_remote.py'`.

Every compile goes through lassi.core.runner.build_toolchain and the
toolchain's own build(), as a run compiles:

- A1, with the real nvcc and nvc++: a generated `#include` of an absolute
  path under $HOME (an existing file there, which the test only names and
  never reads or prints), under the scratch root, and under the runs root (a
  marker header the test writes beside the build dir, in another trial's
  directory) fails with the compiler's missing-file error, and the marker's
  text never reaches stderr; an absolute include of a header in the build
  dir builds, and the pinned compiler's own path is visible to
  __has_include; the layout app of lassi-hecbench-10 (the fetched pinned
  HeCBench sources) still builds with each compiler.
- A2 and A3, with a probe stand-in (a python3 script at the pinned nvcc
  path of a toolchains root the test builds, not a compiler): the compile
  sees exactly PATH, LANG=C, LC_ALL=C, and TMPDIR, no HOME, a TMPDIR that
  is a writable private directory under its build dir, and not a marker
  file in the gate's TMPDIR under the scratch root.
- A5, with a crash stand-in (a /bin/sh script that sends itself SIGSEGV,
  never a real compiler): the compile ends with status 139, the stand-in
  saw a soft core limit of 1 (it refuses to crash otherwise), and
  `coredumpctl list` shows no core for the user's uid since the test began,
  polled for 15 s, as the P0.16 R6 test reads it. The test also refuses to
  crash anything unless the compile's sandbox command starts with
  prlimit --core=1.
- The allowlisted environment itself: a program run through Sandbox.run
  with SandboxSpec.environment sees exactly those variables.
- A4 in the sandbox (review finding): build_toolchain's
  --version check runs through the compile sandbox, so the probe stand-in's
  --version sees the compile environment, a soft core limit of 1, and not a
  marker file in the gate's TMPDIR; and a toolchains root reached through a
  symbolic link still compiles (the compiler is found inside the view).
- The compile output cap (review finding), with a flood stand-in (a python3
  script that prints past COMPILE_OUTPUT_CAP_BYTES on stderr, not a
  compiler): compile.stderr keeps at most the cap and ends with the
  sandbox's line saying the stream was cut.

Each stand-in answers --version with a PLACEHOLDER banner that holds the
cuda pin's EXPECT_VERSION, so build_toolchain's version check passes; the
banner is a test input, not the compiler's. Nothing is written on the host
root filesystem (Agent Rule 7): every file a test writes lies in a temp
directory under $LASSI_SCRATCH/tmp or $LASSI_RUNS_ROOT that it removes
afterwards. No value here is a measurement.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from lassi.bench import Direction, load_suite, sources_dir
from lassi.core.interfaces import BuildResult, Limits
from lassi.core.runner import build_toolchain
from lassi.executors.sandbox import SANDBOX_PATH
from lassi.toolchains import NvccSm80, NvcppCc80
from lassi.toolchains.pins import read_pin

REPO = Path(__file__).resolve().parents[2]
SUITE_MANIFEST = REPO / "assets" / "bench" / "lassi-hecbench-10.yaml"
TOOLS = ("unshare", "systemd-run", "nice", "setpriv", "prlimit", "timeout", "python3", "awk")
REQUIRE = os.environ.get("LASSI_REQUIRE_SANDBOX") == "1"
PRESETS: dict[str, type] = {"nvcc-sm80": NvccSm80, "nvcpp-cc80": NvcppCc80}
SOURCE_NAMES = {"nvcc-sm80": "main.cu", "nvcpp-cc80": "main.cpp"}
# The compiler's missing-file error: GCC (nvcc's host preprocessing) or EDG (nvc++, and nvcc's front end).
NOT_FOUND = re.compile(r"No such file or directory|cannot open source file")
BANNER = "PLACEHOLDER stand-in, not a compiler: " + read_pin("cuda")["EXPECT_VERSION"]
# The stand-ins' exit statuses: the probe reports and fails; the crash stand-in refuses at any other core limit.
PROBE_STATUS = 3
REFUSED_TO_CRASH = 42


def host_problem() -> str:
    """Return why this host cannot run sandboxed compiles, or "" when it can."""
    if not sys.platform.startswith("linux"):
        return f"the sandbox needs Linux, not {sys.platform}"
    missing = [tool for tool in TOOLS if shutil.which(tool) is None]
    if missing:
        return f"not on PATH: {', '.join(missing)}"
    runtime = os.environ.get("XDG_RUNTIME_DIR", "")
    if not runtime or not (Path(runtime) / "bus").exists():
        return "no user systemd manager: $XDG_RUNTIME_DIR/bus does not exist"
    for name in ("LASSI_SCRATCH", "LASSI_RUNS_ROOT", "TMPDIR"):
        if not os.environ.get(name):
            return f"{name} is not set"
    scratch, tmpdir = Path(os.environ["LASSI_SCRATCH"]).resolve(), Path(os.environ["TMPDIR"]).resolve()
    if scratch not in tmpdir.parents:
        return "TMPDIR does not lie inside $LASSI_SCRATCH"
    if shutil.which("python3", path=SANDBOX_PATH) is None:
        return f"no python3 in the sandbox's PATH {SANDBOX_PATH}"
    return ""


def compilers_problem() -> str:
    """Return why the pinned nvcc and nvc++ cannot be used here, or "" when both are installed."""
    root = os.environ.get("LASSI_TOOLCHAINS", "")
    if not root:
        return "LASSI_TOOLCHAINS is not set"
    missing = []
    for preset in PRESETS.values():
        pin = read_pin(preset.PIN)
        executable = Path(root) / pin["PREFIX_NAME"] / preset.PIN_BIN.format_map(pin)
        if not executable.is_file():
            missing.append(str(executable))
    return f"no pinned compiler at {', '.join(missing)}" if missing else ""


PROBLEM = host_problem()
pytestmark = [
    pytest.mark.remote,
    pytest.mark.skipif(bool(PROBLEM) and not REQUIRE, reason=PROBLEM or "the host can run sandboxed compiles"),
]


@pytest.fixture(autouse=True)
def host_ready() -> None:
    """Fail, rather than skip, when LASSI_REQUIRE_SANDBOX=1 asks for real sandboxes and the host cannot run them."""
    if PROBLEM:
        pytest.fail(f"LASSI_REQUIRE_SANDBOX=1, but {PROBLEM}")


def need(condition: bool, why: str) -> None:
    """Skip the test when `condition` is false, or fail it under LASSI_REQUIRE_SANDBOX=1."""
    if not condition:
        if REQUIRE:
            pytest.fail(f"LASSI_REQUIRE_SANDBOX=1, but {why}")
        pytest.skip(why)


def need_compilers() -> Path:
    """Return $LASSI_TOOLCHAINS when the pinned nvcc and nvc++ are installed there; skip or fail otherwise."""
    problem = compilers_problem()
    need(not problem, problem)
    return Path(os.environ["LASSI_TOOLCHAINS"])


@dataclass(frozen=True)
class Trial:
    """A trial-like layout under $LASSI_RUNS_ROOT: this trial's fresh build dir and another trial's directory."""

    base: Path
    workdir: Path
    other: Path


@pytest.fixture
def trial() -> Iterator[Trial]:
    """Create a fresh temp directory under $LASSI_RUNS_ROOT holding two trials; remove it afterwards."""
    base = Path(tempfile.mkdtemp(prefix="lassi-compile-test.", dir=os.environ["LASSI_RUNS_ROOT"]))
    try:
        workdir = base / "trial-a" / "attempt00" / "build"
        workdir.mkdir(parents=True)
        other = base / "trial-b" / "attempt00" / "build"
        other.mkdir(parents=True)
        yield Trial(base=base, workdir=workdir, other=other)
    finally:
        shutil.rmtree(base)


@pytest.fixture
def scratch_dir() -> Iterator[Path]:
    """Create a fresh temp directory under $LASSI_SCRATCH/tmp, outside every build dir; remove it afterwards."""
    parent = Path(os.environ["LASSI_SCRATCH"]) / "tmp"
    parent.mkdir(parents=True, exist_ok=True)
    base = Path(tempfile.mkdtemp(prefix="lassi-compile-test.", dir=parent))
    try:
        yield base
    finally:
        shutil.rmtree(base)


def stand_in_root(base: Path, body: str) -> Path:
    """Write a toolchains root under `base` whose pinned nvcc path holds the executable script `body`; return it."""
    pin = read_pin(NvccSm80.PIN)
    root = base / "toolchains"
    executable = root / pin["PREFIX_NAME"] / NvccSm80.PIN_BIN.format_map(pin)
    executable.parent.mkdir(parents=True)
    executable.write_text(body, encoding="ascii", newline="\n")
    executable.chmod(0o755)
    return root


def build_with(toolchain: str, root: Path, files: dict[str, str], workdir: Path) -> tuple[BuildResult, str]:
    """Build `files` in `workdir` with build_toolchain(toolchain, root); return the result and its stderr."""
    built = build_toolchain(toolchain, root)
    result = built.toolchain.build(files, workdir)
    return result, (workdir / "compile.stderr").read_text(encoding="utf-8")


def marker_header(directory: Path) -> tuple[Path, str]:
    """Write a header holding a fresh marker macro into `directory`; return its path and the marker."""
    token = f"LASSI_MARKER_{uuid.uuid4().hex}"
    path = directory / "secret.h"
    path.write_text(f"#define {token} 1\n", encoding="ascii")
    return path, token


def home_file() -> Path:
    """Return an existing, readable regular file directly under $HOME; its content is never read or printed."""
    home = os.environ.get("HOME", "")
    need(bool(home) and Path(home).is_dir(), "HOME is not set to a directory")
    # On alpha01 HOME is the scratch root itself (both /mnt/nvme10/joseph_ufl), so the file lies under both hidden
    # roots; a file directly under HOME is never inside the build dir or the toolchains root, which lie deeper.
    found = [
        path
        for path in sorted(Path(home).iterdir())
        if path.is_file() and not path.is_symlink() and os.access(path, os.R_OK) and '"' not in path.name
    ]
    need(bool(found), "no readable regular file directly under HOME to include")
    return found[0]


# ---------------------------------------------------------------------------
# The allowlisted environment reaches the program exactly


def test_a_program_sees_exactly_the_allowlisted_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from lassi.executors.sandbox import Sandbox, SandboxSpec

    secret = f"lassi-compile-test-secret-{uuid.uuid4().hex}"
    monkeypatch.setenv("LASSI_TEST_API_KEY", secret)
    workdir = tmp_path / "build"
    workdir.mkdir()
    environment = {"PATH": SANDBOX_PATH, "LANG": "C", "LC_ALL": "C", "TMPDIR": str(workdir / "tmp")}
    spec = SandboxSpec(
        workdir=workdir.resolve(),
        hidden_roots=(Path(os.environ["LASSI_SCRATCH"]).resolve(),),
        environment=environment,
    )
    program = shutil.which("env", path=SANDBOX_PATH)
    need(program is not None, f"no env in {SANDBOX_PATH}")
    result = Sandbox().run(spec, [str(program)], Limits(wall_s=10.0, memory_mb=64, cpus=1))
    assert result.returncode == 0, result
    assert sorted(result.stdout.splitlines()) == sorted(f"{name}={value}" for name, value in environment.items())
    assert secret not in result.stdout + result.stderr


# ---------------------------------------------------------------------------
# A2 and A3: a probe stand-in reports the compile's environment and view

PROBE = """#!{python} -I
import json, os, sys
def readable(path):
    try:
        with open(path, "rb") as handle:
            handle.read(1)
        return "readable"
    except OSError as exc:
        return "error:%d" % exc.errno
if sys.argv[1:] == ["--version"]:
    print({banner!r})
    with open("/proc/self/limits") as handle:
        core = [line.split()[4] for line in handle if line.startswith("Max core file size")][0]
    print("LASSI_VERSION_VIEW env=%s core=%s marker=%s" % (",".join(sorted(os.environ)), core, readable({marker!r})))
    sys.exit(0)
report = {{"environ": dict(os.environ), "marker": readable({marker!r})}}
try:
    with open(os.path.join(os.environ.get("TMPDIR", "/nonexistent"), "probe.txt"), "w") as handle:
        handle.write("probe\\n")
    report["tmpdir_write"] = "ok"
except OSError as exc:
    report["tmpdir_write"] = "error:%d" % exc.errno
sys.stderr.write("LASSI_PROBE " + json.dumps(report) + "\\n")
sys.exit({status})
"""


def probe_report(stderr: str) -> dict[str, object]:
    """Return the JSON report the probe stand-in printed on its LASSI_PROBE line."""
    lines = [line for line in stderr.splitlines() if line.startswith("LASSI_PROBE ")]
    assert len(lines) == 1, stderr
    return json.loads(lines[0][len("LASSI_PROBE ") :])


def test_a2_a3_a_compile_sees_its_environment_and_its_private_tmpdir_and_not_the_rest_of_scratch(
    scratch_dir: Path,
) -> None:
    python = shutil.which("python3", path=SANDBOX_PATH)
    assert python is not None
    gate_tmp = Path(os.environ["TMPDIR"])
    handle, marker = tempfile.mkstemp(prefix="lassi-compile-test-marker.", dir=gate_tmp)
    os.close(handle)
    try:
        body = PROBE.format(python=python, banner=BANNER, marker=marker, status=PROBE_STATUS)
        root = stand_in_root(scratch_dir, body)
        workdir = scratch_dir / "build"
        workdir.mkdir()
        result, stderr = build_with("nvcc-sm80", root, {"main.cu": "int main() { return 0; }\n"}, workdir)
    finally:
        os.unlink(marker)
    report = probe_report(stderr)
    environ = report["environ"]
    assert isinstance(environ, dict)
    assert sorted(environ) == ["LANG", "LC_ALL", "PATH", "TMPDIR"], sorted(environ)
    assert (environ["LANG"], environ["LC_ALL"], environ["PATH"]) == ("C", "C", os.environ["PATH"])
    tmpdir = Path(str(environ["TMPDIR"]))
    assert workdir.resolve() in tmpdir.resolve().parents, tmpdir
    assert tmpdir.resolve() != gate_tmp.resolve()
    assert report["tmpdir_write"] == "ok", report
    assert str(report["marker"]).startswith("error:"), report
    assert result.artifact is None


def test_a4_the_version_check_runs_in_the_compile_sandbox(scratch_dir: Path) -> None:
    python = shutil.which("python3", path=SANDBOX_PATH)
    assert python is not None
    handle, marker = tempfile.mkstemp(prefix="lassi-compile-test-marker.", dir=os.environ["TMPDIR"])
    os.close(handle)
    try:
        body = PROBE.format(python=python, banner=BANNER, marker=marker, status=PROBE_STATUS)
        built = build_toolchain("nvcc-sm80", stand_in_root(scratch_dir, body))
    finally:
        os.unlink(marker)
    views = [line for line in built.version if line.startswith("LASSI_VERSION_VIEW ")]
    assert len(views) == 1, built.version
    fields = dict(item.split("=", 1) for item in views[0].split()[1:])
    assert fields["env"] == "LANG,LC_ALL,PATH,TMPDIR", views
    assert fields["core"] == "1", "the check runs under prlimit --core=1"
    assert fields["marker"].startswith("error:"), "the check runs in the compile view, which hides the gate's TMPDIR"
    assert built.version_status == 0


def test_f2_a_toolchains_root_behind_a_symbolic_link_still_compiles(scratch_dir: Path) -> None:
    # The sandbox exposes the resolved root, so a compile through the link must name the compiler by that root;
    # otherwise env inside would not find it and report 127 as the compiler's status (review finding).
    python = shutil.which("python3", path=SANDBOX_PATH)
    assert python is not None
    body = PROBE.format(python=python, banner=BANNER, marker="/nonexistent", status=PROBE_STATUS)
    real = stand_in_root(scratch_dir, body)
    link = scratch_dir / "toolchains-link"
    link.symlink_to(real, target_is_directory=True)
    workdir = scratch_dir / "build"
    workdir.mkdir()
    result, stderr = build_with("nvcc-sm80", link, {"main.cu": "int main() { return 0; }\n"}, workdir)
    report = probe_report(stderr)
    assert report["tmpdir_write"] == "ok", report
    assert [d.code for d in result.diagnostics if f"status {PROBE_STATUS} " in d.message] == ["exit-status"]


# A stand-in that prints `size` bytes of long lines on stderr (no diagnostic form, so parsing stays cheap), then
# exits 1; --version prints the banner.
FLOOD = """#!{python} -I
import sys
if sys.argv[1:] == ["--version"]:
    print({banner!r})
    sys.exit(0)
block = (b"lassi flood line " + b"x" * 1006 + b"\\n") * 64
written = 0
while written < {size}:
    sys.stderr.buffer.write(block)
    written += len(block)
sys.exit(1)
"""


def test_f1_a_compiler_flood_is_cut_at_the_compile_output_cap_and_says_so(scratch_dir: Path) -> None:
    from lassi.executors.sandbox import COMPILE_OUTPUT_CAP_BYTES

    python = shutil.which("python3", path=SANDBOX_PATH)
    assert python is not None
    size = COMPILE_OUTPUT_CAP_BYTES + (8 << 20)
    root = stand_in_root(scratch_dir, FLOOD.format(python=python, banner=BANNER, size=size))
    workdir = scratch_dir / "build"
    workdir.mkdir()
    result, _stderr = build_with("nvcc-sm80", root, {"main.cu": "int main() { return 0; }\n"}, workdir)
    kept = (workdir / "compile.stderr").read_bytes()
    assert COMPILE_OUTPUT_CAP_BYTES - 8192 < len(kept) <= COMPILE_OUTPUT_CAP_BYTES + 1024, len(kept)
    assert kept.startswith(b"lassi flood line "), kept[:80]
    last = kept.rstrip(b"\n").rsplit(b"\n", 1)[-1]
    assert last.startswith(b"lassi-sandbox:") and str(COMPILE_OUTPUT_CAP_BYTES).encode("ascii") in last, last
    assert result.artifact is None
    assert [d.code for d in result.diagnostics if "status 1 " in d.message] == ["exit-status"], result.diagnostics


# ---------------------------------------------------------------------------
# A5: a crash stand-in stores no core with the host handler

CRASH = """#!/bin/sh
if [ "$1" = "--version" ]; then
  printf '%s\\n' '{banner}'
  exit 0
fi
set -- $(grep '^Max core file size' /proc/self/limits)
if [ "$5" != 1 ]; then
  echo "lassi-crash-stand-in: soft core limit $5, not 1; refusing to crash" >&2
  exit {refused}
fi
echo "lassi-crash-stand-in: soft core limit 1; crashing" >&2
kill -SEGV $$
echo "lassi-crash-stand-in: still running after SIGSEGV" >&2
exit 1
"""


def test_a5_a_compiler_crash_stores_no_core_with_the_host_handler(scratch_dir: Path) -> None:
    from lassi.executors.sandbox import SandboxedCompileRunner, sandbox_command

    need(shutil.which("coredumpctl") is not None, "coredumpctl is not on PATH, so the host handler cannot be read")
    assert "'" not in BANNER
    root = stand_in_root(scratch_dir, CRASH.format(banner=BANNER, refused=REFUSED_TO_CRASH))
    workdir = scratch_dir / "build"
    workdir.mkdir()
    built = build_toolchain("nvcc-sm80", root)
    runner = built.toolchain.runner
    assert isinstance(runner, SandboxedCompileRunner)
    # Refuse to crash anything unless the compile runs under the core limit of 1: at 0 the host handler stores
    # the core on the root filesystem (plans/spikes/p0-sandbox-hardening.md, probe G1; Agent Rule 7).
    command = sandbox_command(runner.spec(workdir), ["true"], runner.limits(built.toolchain.timeout_s))
    assert command[:3] == ["prlimit", "--core=1", "--"], command[:6]
    since = int(time.time()) - 1
    result = built.toolchain.build({"main.cu": "int main() { return 0; }\n"}, workdir)
    stderr = (workdir / "compile.stderr").read_text(encoding="utf-8")
    assert "soft core limit 1; crashing" in stderr, stderr
    assert "still running" not in stderr, stderr
    assert result.artifact is None
    assert [d.code for d in result.diagnostics if "status 139" in d.message] == ["exit-status"], result.diagnostics
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(since))
    listing_command = ["coredumpctl", "--no-pager", "list", f"--since={stamp}", f"COREDUMP_UID={os.getuid()}"]
    # systemd-coredump works asynchronously, so the journal is read every second for 15 s, and any entry fails.
    deadline = time.monotonic() + 15
    while True:
        listing = subprocess.run(listing_command, capture_output=True, text=True, timeout=60, check=False)
        assert listing.stdout.strip() == "", listing
        if time.monotonic() >= deadline:
            break
        time.sleep(1)
    assert listing.returncode == 1 and listing.stdout.strip() == "", listing
    assert "No coredumps found" in listing.stderr and "permission" not in listing.stderr.lower(), listing
    assert not list(workdir.glob("core*"))


# ---------------------------------------------------------------------------
# A1: the real pinned compilers


@pytest.mark.slow
@pytest.mark.parametrize("where", ["home", "scratch-root", "runs-root"])
@pytest.mark.parametrize("toolchain", sorted(PRESETS))
def test_a1_a_generated_include_of_a_hidden_absolute_path_fails(
    toolchain: str, where: str, trial: Trial, scratch_dir: Path
) -> None:
    root = need_compilers()
    token = ""
    if where == "home":
        included = home_file()
    elif where == "scratch-root":
        included, token = marker_header(scratch_dir)
    else:
        included, token = marker_header(trial.other)
    assert '"' not in str(included)
    source = f'#include "{included}"\nint main() {{ return 0; }}\n'
    result, stderr = build_with(toolchain, root, {SOURCE_NAMES[toolchain]: source}, trial.workdir)
    # Never print stderr for the home case: a broken sandbox would have echoed the file's lines into it.
    facts = {
        "artifact": result.artifact is not None,
        "codes": sorted({d.code for d in result.diagnostics}),
        "not_found": bool(NOT_FOUND.search(stderr)),
        "names_the_path": str(included) in stderr,
        "stderr_bytes": len(stderr.encode("utf-8")),
    }
    shown = facts if where == "home" else {**facts, "stderr": stderr}
    assert result.artifact is None, shown
    assert facts["not_found"] and facts["names_the_path"], shown
    if token:
        assert token not in stderr, shown


@pytest.mark.slow
@pytest.mark.parametrize("toolchain", sorted(PRESETS))
def test_a1_the_build_dir_and_the_pinned_toolchains_stay_readable(toolchain: str, trial: Trial) -> None:
    root = need_compilers()
    executable = build_toolchain(toolchain, root).executable
    assert executable is not None and '"' not in executable
    local = trial.workdir.resolve() / "local.h"
    source = (
        f'#include "{local}"\n'
        f'#if !__has_include("{executable}")\n'
        "#error LASSI_TOOLCHAINS_HIDDEN\n"
        "#endif\n"
        "int main() { return lassi_local_value; }\n"
    )
    files = {SOURCE_NAMES[toolchain]: source, "local.h": "static int lassi_local_value = 0;\n"}
    result, stderr = build_with(toolchain, root, files, trial.workdir)
    assert "LASSI_TOOLCHAINS_HIDDEN" not in stderr, stderr
    assert result.artifact is not None and result.artifact.is_file(), (result.diagnostics, stderr)


@pytest.mark.slow
@pytest.mark.parametrize(("toolchain", "target"), [("nvcc-sm80", "cuda"), ("nvcpp-cc80", "omp")])
def test_a1_a_hecbench_sized_compile_still_builds(toolchain: str, target: str, trial: Trial) -> None:
    # The layout app's reference target in the toolchain's language, from the fetched pinned HeCBench sources:
    # the compile limits must fit it. Compiling a reference is no training or tuning on the eval item (Rule 5).
    root = need_compilers()
    suite = load_suite(SUITE_MANIFEST)
    sources = sources_dir(Path(os.environ["LASSI_SCRATCH"]), suite)
    need(sources.is_dir(), f"the {suite.name} sources are not fetched under {sources}; run tools/fetch_bench.py")
    direction = Direction(source="omp" if target == "cuda" else "cuda", target=target)
    files = suite.reference_target("layout", direction, sources, purpose="eval")
    result, stderr = build_with(toolchain, root, files, trial.workdir)
    assert result.artifact is not None and result.artifact.is_file(), (result.diagnostics, stderr)
