"""Tests for tools/capture_toolchain_fixtures.py and the scenario manifest it reads (P0.15, stage A).

The capture tool compiles each scenario's hand-written sources,
tests/toolchains/fixtures/sources/<scenario>/, with the pinned toolchain built
exactly as the stage runner builds it (lassi.core.runner.build_toolchain: the
pinned executable, the clean compile environment, linked prefixes such as
NVHPC_CUDA_HOME, and EnvRunner). Each scenario builds in a fresh workdir,
<out>/work/<scenario>/, and the tool copies the compiler's stderr byte for
byte to <out>/<scenario>.stderr and writes the provenance manifest
<out>/manifest.json (bible Component Interfaces, Toolchain contract rules;
Result Record, Diagnostic; Agent Rules 1, 7, and 10).
tests/toolchains/fixtures/scenarios.json names each scenario's toolchain, an
optional ARCH or GPU override of the preset, and a one-line description; its
names are those of the .stderr fixtures.

The manifest layout these tests pin (the interface the tool follows):

- top level: date (ISO 8601 with an offset), commit, dirty, snapshot_of (the
  commit an rx snapshot HEAD was made from, else null), rx_run_id
  (LASSI_RX_RUN_ID or null), host (platform.node()), toolchains, scenarios;
- toolchains.<registry name>: pins (pin name -> VERSION), executable,
  environment (the sorted variable names, never their values), locale
  ({"LANG": "C", "LC_ALL": "C"}, the only values shown), version (the
  non-blank lines of `<executable> --version`, in order), version_exit_status,
  and pin_files (pin name -> every pair of its pin file); the tool refuses
  to capture when --version fails or does not print the pin's EXPECT_VERSION;
- scenarios.<scenario>: toolchain, overrides (class attribute -> value, empty
  for the preset), argv (as the adapter ran it; the sources are relative to
  the workdir), exit_status, stderr_sha256, stderr_bytes, and diagnostics (how
  many Diagnostics the adapter's parse() finds in the stderr).

No test here runs a compiler. The toolchains root is a temporary directory
holding empty placeholder files at the pinned executable paths, and
subprocess.Popen is replaced: git runs for real, a pinned executable gets a
canned reply (for --version, a PLACEHOLDER banner holding the pin's
EXPECT_VERSION, and for a compile the current .stderr fixture of the
scenario it builds), and any other command fails the test, so no built
program can run either. The one `remote` test runs
the tool with the real pinned compilers on the build host and skips
elsewhere. No value in this module is a measurement.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
import os
import platform
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any

import pytest

from lassi.core.registry import DEFAULT_REGISTRY
from lassi.toolchains import NvccSm80, NvcppCc80
from lassi.toolchains.pins import read_pin

REPO = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO / "tools"
TOOL = TOOLS_DIR / "capture_toolchain_fixtures.py"
TOOL_MODULE = "capture_toolchain_fixtures"
FIXTURES = REPO / "tests" / "toolchains" / "fixtures"
SCENARIOS = FIXTURES / "scenarios.json"
SOURCES = FIXTURES / "sources"
# The fixture names today; the scenario manifest and the source trees follow them.
FIXTURE_NAMES = sorted(path.stem for path in FIXTURES.glob("*.stderr"))

# The pinned executables under a toolchains root as toolchains/cuda.pin and toolchains/nvhpc.pin name them,
# and the CUDA prefix nvc++ gets as NVHPC_CUDA_HOME (as in tests/core/test_runner.py).
NVCC_BIN = "cuda@12.6.3/bin/nvcc"
NVCPP_BIN = "nvhpc@24.11/Linux_x86_64/24.11/compilers/bin/nvc++"
CUDA_PREFIX = "cuda@12.6.3"

# Per toolchain: the preset class, the fixture name prefix, the pinned executable, the pin versions, and the
# scenarios.json key and class attribute of its one allowed override.
PRESETS: dict[str, type] = {"nvcc-sm80": NvccSm80, "nvcpp-cc80": NvcppCc80}
NAME_PREFIXES = {"nvcc-sm80": "nvcc_", "nvcpp-cc80": "nvcpp_"}
BINARIES = {"nvcc-sm80": NVCC_BIN, "nvcpp-cc80": NVCPP_BIN}
PIN_VERSIONS = {"nvcc-sm80": {"cuda": "12.6.3"}, "nvcpp-cc80": {"nvhpc": "24.11", "cuda": "12.6.3"}}
OVERRIDES = {"nvcc-sm80": ("arch", "ARCH"), "nvcpp-cc80": ("gpu", "GPU")}
ENTRY_KEYS = frozenset({"toolchain", "arch", "gpu", "scenario"})

# The names build() writes into the workdir itself, so no source may use them as a first segment.
RESERVED = ("main", "compile.stderr")
# One segment of a source path: tame names only, so every path is a plain relative POSIX path.
SEGMENT = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")

MANIFEST_KEYS = frozenset({"date", "commit", "dirty", "snapshot_of", "rx_run_id", "host", "toolchains", "scenarios"})
TOOLCHAIN_KEYS = frozenset(
    {"pins", "pin_files", "executable", "environment", "locale", "version", "version_exit_status"}
)
SCENARIO_KEYS = frozenset(
    {"toolchain", "overrides", "argv", "exit_status", "stderr_sha256", "stderr_bytes", "diagnostics"}
)
# The subject tools/rx.py gives the snapshot commit it sends for a dirty working tree.
SNAPSHOT_PREFIX = "rx snapshot of "
# An identity for commits in a throwaway repository, from the environment only.
IDENTITY = {
    "GIT_AUTHOR_NAME": "Capture Test",
    "GIT_AUTHOR_EMAIL": "capture-test@example.invalid",
    "GIT_COMMITTER_NAME": "Capture Test",
    "GIT_COMMITTER_EMAIL": "capture-test@example.invalid",
}
LOCALE = {"LANG": "C", "LC_ALL": "C"}

# What the fake --version prints after its pinned line (pinned_line): a PLACEHOLDER banner with blank lines,
# which the manifest leaves out.
VERSION_BANNER = (
    b"\nPLACEHOLDER compiler banner line 1\nPLACEHOLDER line 2\n\nPLACEHOLDER line 3\n"
    b"PLACEHOLDER line 4\nPLACEHOLDER line 5\n"
)
VERSION_LINES = [
    "PLACEHOLDER compiler banner line 1",
    "PLACEHOLDER line 2",
    "PLACEHOLDER line 3",
    "PLACEHOLDER line 4",
    "PLACEHOLDER line 5",
]
# The exit status the fake compiler gives a scenario whose stderr parses into an error.
FAILED = 2
# Stderr that a plain-text copy would change: UTF-8 quotes (as GCC prints them outside the C locale),
# a CR LF line end, and no final newline. Its capture must keep every byte.
ODD_STDERR = b"PLACEHOLDER stderr with UTF-8 quotes \xe2\x80\x98x\xe2\x80\x99\r\nand a last line without a newline"
RX_RUN_ID = "rx-test-0001"


# Distinct parts of the values of HOME, TMPDIR, and PATH, which must never reach the manifest.
HOME_MARKER = "home-value-marker"
TMPDIR_MARKER = "tmpdir-value-marker"
PATH_MARKER = "path-value-marker"

ALLOWED_THIRD_PARTY = frozenset({"yaml", "pyarrow"})
FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)
# Pin resolution belongs to lassi.core.runner; the tool calls build_toolchain instead of repeating it.
DUPLICATED_LOGIC = frozenset(
    {"read_pin", "linked_prefixes", "prefix_pin_name", "_compile_environment", "_pinned_toolchain"}
)


def pinned_line(toolchain: str) -> str:
    """Return the first line the fake `--version` prints: a PLACEHOLDER holding the pin's EXPECT_VERSION."""
    return "PLACEHOLDER banner holding " + read_pin(PRESETS[toolchain].PIN)["EXPECT_VERSION"]


# ---------------------------------------------------------------------------
# The scenario manifest and the source trees


def load_scenarios() -> dict[str, dict[str, Any]]:
    """Return tests/toolchains/fixtures/scenarios.json, after checking it is plain ASCII JSON with LF newlines."""
    raw = SCENARIOS.read_bytes()
    assert raw.isascii(), f"{SCENARIOS} is not plain ASCII"
    assert b"\r" not in raw, f"{SCENARIOS} has a CR"
    data = json.loads(raw.decode("ascii"))
    assert isinstance(data, dict), f"{SCENARIOS} must hold one JSON object, scenario name -> entry"
    for name, entry in data.items():
        assert isinstance(name, str) and isinstance(entry, dict), name
    return data


def source_bytes(name: str) -> dict[str, bytes]:
    """Return the files of the scenario's source tree, relative POSIX path -> bytes, in sorted order."""
    tree = SOURCES / name
    return {path.relative_to(tree).as_posix(): path.read_bytes() for path in sorted(tree.rglob("*")) if path.is_file()}


def source_texts(name: str) -> dict[str, str]:
    """Return the files of the scenario's source tree as the adapter's build() takes them: path -> text."""
    return {path: data.decode("utf-8") for path, data in source_bytes(name).items()}


def adapter_class(entry: Mapping[str, Any]) -> type:
    """Return the class the tool builds for a scenario: the preset, or a subclass that sets only its override."""
    preset = PRESETS[entry["toolchain"]]
    key, attribute = OVERRIDES[entry["toolchain"]]
    if key not in entry:
        return preset
    return type(f"{preset.__name__}Override", (preset,), {attribute: entry[key]})


def expected_overrides(entry: Mapping[str, Any]) -> dict[str, str]:
    """Return the class attributes the scenario overrides, attribute -> value; empty for the preset."""
    key, attribute = OVERRIDES[entry["toolchain"]]
    return {attribute: entry[key]} if key in entry else {}


def expected_argv(entry: Mapping[str, Any], name: str, executable: str) -> list[str]:
    """Return the command the adapter runs for the scenario: its own command() over the sorted sources."""
    cls = adapter_class(entry)
    sources = sorted(path for path in source_bytes(name) if PurePosixPath(path).suffix in cls.SOURCE_SUFFIXES)
    return cls(executable=executable).command(sources)


def parsed_count(entry: Mapping[str, Any], name: str, stderr: bytes) -> int:
    """Return how many Diagnostics the scenario's adapter parses from `stderr` with its source files."""
    return len(PRESETS[entry["toolchain"]]().parse(stderr.decode("utf-8", errors="replace"), source_texts(name)))


def test_scenario_names_match_the_stderr_fixtures() -> None:
    assert FIXTURE_NAMES, f"no .stderr fixtures in {FIXTURES}"
    assert sorted(load_scenarios()) == FIXTURE_NAMES


def test_the_source_trees_are_exactly_the_scenarios() -> None:
    trees = sorted(path.name for path in SOURCES.iterdir())
    assert trees == sorted(load_scenarios()), "every entry under sources/ is one scenario's directory"
    assert all((SOURCES / name).is_dir() for name in trees)


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_scenario_names_its_toolchain_and_one_line(name: str) -> None:
    entry = load_scenarios()[name]
    assert set(entry) <= ENTRY_KEYS, f"{name}: unknown keys {sorted(set(entry) - ENTRY_KEYS)}"
    assert entry.get("toolchain") in PRESETS, f"{name}: toolchain must be one of {sorted(PRESETS)}"
    assert entry["toolchain"] in DEFAULT_REGISTRY.names("Toolchain")
    assert name.startswith(NAME_PREFIXES[entry["toolchain"]]), f"{name} does not match its toolchain"
    description = entry.get("scenario")
    assert isinstance(description, str) and description.strip(), f"{name}: no scenario description"
    assert description.isascii() and "\n" not in description and "\r" not in description, name


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_scenario_overrides_only_its_presets_attribute(name: str) -> None:
    entry = load_scenarios()[name]
    key, attribute = OVERRIDES[entry["toolchain"]]
    other = {candidate for candidate, _ in OVERRIDES.values()} - {key}
    assert not other & set(entry), f"{name}: {entry['toolchain']} has no {sorted(other & set(entry))} override"
    if key not in entry:
        return
    value = entry[key]
    assert isinstance(value, str) and value and value.isascii() and not any(ch.isspace() for ch in value), name
    preset = PRESETS[entry["toolchain"]]
    assert value != getattr(preset, attribute), f"{name}: an override equal to the preset's {attribute} is not one"


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_every_scenario_has_a_source_tree_of_relative_ascii_files(name: str) -> None:
    entry = load_scenarios()[name]
    tree = SOURCES / name
    assert tree.is_dir(), f"no source tree {tree}"
    links = [path for path in tree.rglob("*") if path.is_symlink()]
    assert not links, f"{name}: links in the source tree: {links}"
    files = source_bytes(name)
    assert files, f"{name}: the source tree is empty"
    for path, data in files.items():
        segments = path.split("/")
        assert all(SEGMENT.fullmatch(segment) for segment in segments), f"{name}: {path!r} is not a plain path"
        assert not PurePosixPath(path).is_absolute() and ".." not in segments, f"{name}: {path!r}"
        assert segments[0] not in RESERVED, f"{name}: {path!r} uses a name build() writes itself"
        assert data.isascii(), f"{name}: {path} is not plain ASCII"
        assert b"\r" not in data, f"{name}: {path} has a CR"
    suffixes = PRESETS[entry["toolchain"]].SOURCE_SUFFIXES
    compiled = [path for path in files if PurePosixPath(path).suffix in suffixes]
    assert compiled, f"{name}: no file the {entry['toolchain']} adapter compiles ({', '.join(suffixes)})"


# ---------------------------------------------------------------------------
# A fake host: pinned placeholder executables and a subprocess.Popen that starts no compiler


@dataclass(frozen=True)
class Spawned:
    """One command the fake subprocess.Popen was asked to start: argv, working directory, and environment."""

    argv: list[str]
    cwd: Path | None
    env: dict[str, str] | None


@dataclass(frozen=True)
class Reply:
    """What the fake compiler returns for one scenario's compile: the exit status and the stderr bytes."""

    status: int
    stderr: bytes


class FinishedProcess:
    """What the fake subprocess.Popen returns: a process that has already exited; nothing was started."""

    # A pid no process has, so a kill (only after a timeout, which never happens here) finds nothing.
    pid = 2**31 - 1

    def __init__(self, argv: list[str], returncode: int, output: tuple[bytes, bytes], text: bool) -> None:
        """Keep the exit status and the output, as text when the caller asked for text."""
        self.args = argv
        self.returncode = returncode
        self.stdin = self.stdout = self.stderr = None
        decoded = tuple(data.decode("utf-8", errors="replace") for data in output)
        self._output: tuple[Any, Any] = decoded if text else output

    def communicate(self, input: Any = None, timeout: float | None = None) -> tuple[Any, Any]:
        """Return the stdout and stderr the fake was given."""
        return self._output

    def poll(self) -> int:
        """Return the exit status."""
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        """Return the exit status."""
        return self.returncode

    def kill(self) -> None:
        """Do nothing: no process was started."""

    def terminate(self) -> None:
        """Do nothing: no process was started."""

    def send_signal(self, signal: int) -> None:
        """Do nothing: no process was started."""

    def __enter__(self) -> FinishedProcess:
        """Return self, as Popen does."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Do nothing on exit."""


@dataclass
class FakeCompilers:
    """Stands in for subprocess.Popen: git runs for real, a pinned executable under `root` gets a canned reply.

    `<executable> --version` prints pinned_line, then VERSION_BANNER, and
    exits 0, unless `banners` or `version_status` (keyed by toolchain) say
    otherwise. A compile runs in the
    scenario's workdir, whose name is the scenario, and returns that
    scenario's Reply; a compile that exits 0 leaves a placeholder `main`, as
    a compiler that succeeded would. Any other command fails the test, so no
    real compiler and no built program ever runs.
    """

    root: Path
    replies: dict[str, Reply] = field(default_factory=dict)
    calls: list[Spawned] = field(default_factory=list)
    banners: dict[str, bytes] = field(default_factory=dict)
    version_status: dict[str, int] = field(default_factory=dict)

    def pinned(self, executable: str) -> bool:
        """Return True when `executable` lies under the fake toolchains root."""
        path = os.path.normcase(os.path.abspath(executable))
        return path.startswith(os.path.normcase(os.path.abspath(self.root)) + os.sep)

    def compiles(self) -> list[Spawned]:
        """Return the compile commands, in order."""
        return [call for call in self.calls if self.pinned(call.argv[0]) and "--version" not in call.argv[1:]]

    def versions(self) -> list[Spawned]:
        """Return the --version commands, in order."""
        return [call for call in self.calls if self.pinned(call.argv[0]) and "--version" in call.argv[1:]]

    def toolchain_of(self, executable: str) -> str:
        """Return the toolchain whose pinned executable `executable` is."""
        return next(name for name, binary in BINARIES.items() if Path(executable) == self.root / binary)

    def spawn(self, argv: list[str], kwargs: Mapping[str, Any]) -> FinishedProcess:
        """Record one command and return a finished process with its canned reply."""
        cwd = None if kwargs.get("cwd") is None else Path(kwargs["cwd"])
        env = None if kwargs.get("env") is None else dict(kwargs["env"])
        self.calls.append(Spawned(argv=argv, cwd=cwd, env=env))
        text = any(kwargs.get(key) for key in ("text", "universal_newlines", "encoding", "errors"))
        if not self.pinned(argv[0]):
            raise AssertionError(f"the capture tool started {argv!r}, which is neither git nor a pinned compiler")
        if "--version" in argv[1:]:
            toolchain = self.toolchain_of(argv[0])
            default = (pinned_line(toolchain) + "\n").encode("ascii") + VERSION_BANNER
            banner = self.banners.get(toolchain, default)
            return FinishedProcess(argv, self.version_status.get(toolchain, 0), (banner, b""), text)
        assert cwd is not None, f"a compile without a workdir: {argv!r}"
        reply = self.replies.get(cwd.name)
        assert reply is not None, f"a compile in {cwd}, which is no scenario's workdir"
        if reply.status == 0:
            (cwd / "main").write_bytes(b"PLACEHOLDER artifact of a faked compile\n")
        return FinishedProcess(argv, reply.status, (b"", reply.stderr), text)


@dataclass(frozen=True)
class Host:
    """The fake build host of one test: its toolchains root, runs root, compile environments, and fake Popen."""

    root: Path
    runs_root: Path
    compile_env: dict[str, dict[str, str]]
    fake: FakeCompilers


@pytest.fixture
def host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Host:
    """Set os.environ as the gate sets it on the build host, with placeholder pinned compilers and a fake Popen.

    HOME, TMPDIR, and PATH carry marker names, the locale is not C, and
    variables that change a compile silently are set; the compile
    environment is PATH, LANG=C, LC_ALL=C, HOME, and TMPDIR, plus
    NVHPC_CUDA_HOME for nvc++.
    """
    for name in ("LASSI_SCRATCH", "LASSI_RX_RUN_ID", "CPATH"):
        monkeypatch.delenv(name, raising=False)
    root = tmp_path / "toolchains"
    (root / CUDA_PREFIX).mkdir(parents=True)
    for relative in BINARIES.values():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    runs_root, home, tmpdir = tmp_path / "runs", tmp_path / HOME_MARKER, tmp_path / TMPDIR_MARKER
    for directory in (runs_root, home, tmpdir):
        directory.mkdir()
    monkeypatch.setenv("PATH", str(tmp_path / PATH_MARKER) + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("TMPDIR", str(tmpdir))
    monkeypatch.setenv("LANG", "de_DE.UTF-8")
    monkeypatch.setenv("LC_ALL", "de_DE.UTF-8")
    monkeypatch.setenv("NVCC_PREPEND_FLAGS", "-DLEAKED_PREPEND")
    monkeypatch.setenv("NVCC_APPEND_FLAGS", "-DLEAKED_APPEND")
    monkeypatch.setenv("LASSI_TOOLCHAINS", str(root))
    monkeypatch.setenv("LASSI_RUNS_ROOT", str(runs_root))
    base = {"PATH": os.environ["PATH"], "LANG": "C", "LC_ALL": "C", "HOME": str(home), "TMPDIR": str(tmpdir)}
    compile_env = {"nvcc-sm80": base, "nvcpp-cc80": {**base, "NVHPC_CUDA_HOME": str(root / CUDA_PREFIX)}}
    fake = FakeCompilers(root)
    real_popen = subprocess.Popen
    # platform.node() may start a command once (`ver` on Windows) and caches the answer; ask before the fake.
    platform.node()

    def popen(args: Any, *more: Any, **kwargs: Any) -> Any:
        items = [args] if isinstance(args, (str, bytes, os.PathLike)) else list(args)
        argv = [os.fsdecode(item) for item in items]
        if Path(argv[0]).stem.lower() == "git":
            return real_popen(args, *more, **kwargs)
        return fake.spawn(argv, kwargs)

    monkeypatch.setattr(subprocess, "Popen", popen)
    return Host(root=root, runs_root=runs_root, compile_env=compile_env, fake=fake)


@pytest.fixture
def tool() -> ModuleType:
    """Return the capture tool, imported from tools/ as the other tool tests import theirs."""
    if str(TOOLS_DIR) not in sys.path:
        sys.path.insert(0, str(TOOLS_DIR))
    return importlib.import_module(TOOL_MODULE)


@dataclass(frozen=True)
class ToolRun:
    """How one call of the tool's main() ended: its status and what it printed."""

    status: int
    out: str
    err: str


def run_tool(tool: ModuleType, argv: Sequence[str], capsys: pytest.CaptureFixture[str]) -> ToolRun:
    """Call the tool's main(argv) and return its status (a SystemExit counts) and its printed output."""
    try:
        status: Any = tool.main(list(argv))
    except SystemExit as exit_:
        status = exit_.code
    captured = capsys.readouterr()
    if isinstance(status, str):
        return ToolRun(1, captured.out, captured.err + status)
    return ToolRun(0 if status is None else int(status), captured.out, captured.err)


def fixture_replies(scenarios: Mapping[str, Mapping[str, Any]]) -> dict[str, Reply]:
    """Return a Reply per scenario: its current .stderr fixture, with FAILED when that parses into an error."""
    replies: dict[str, Reply] = {}
    for name, entry in scenarios.items():
        stderr = (FIXTURES / f"{name}.stderr").read_bytes()
        diagnostics = PRESETS[entry["toolchain"]]().parse(stderr.decode("utf-8"), source_texts(name))
        failed = any(diagnostic.severity == "error" for diagnostic in diagnostics)
        replies[name] = Reply(FAILED if failed else 0, stderr)
    return replies


def read_manifest(out: Path) -> dict[str, Any]:
    """Return <out>/manifest.json, after checking it is plain ASCII JSON."""
    raw = (out / "manifest.json").read_bytes()
    assert raw.isascii(), "manifest.json must be plain ASCII (stage B keeps it in the repository)"
    data = json.loads(raw.decode("ascii"))
    assert isinstance(data, dict)
    return data


def git_output(*args: str) -> str:
    """Return the stdout of `git <args>` run in the repository."""
    done = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=60, check=True)
    return done.stdout


def strings_in(value: Any) -> list[str]:
    """Return every string in a JSON value, keys included."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for key, item in value.items() for text in [key, *strings_in(item)]]
    if isinstance(value, list):
        return [text for item in value for text in strings_in(item)]
    return []


@dataclass(frozen=True)
class Capture:
    """One full capture with the fake host: the out dir, the tool's run, the manifest, and what was expected."""

    out: Path
    run: ToolRun
    manifest: dict[str, Any]
    scenarios: dict[str, dict[str, Any]]
    replies: dict[str, Reply]
    started: datetime
    finished: datetime


@pytest.fixture
def capture(
    tool: ModuleType, host: Host, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> Capture:
    """Run the tool over every scenario into <tmp>/out with LASSI_RX_RUN_ID set; the first scenario gets ODD_STDERR."""
    monkeypatch.setenv("LASSI_RX_RUN_ID", RX_RUN_ID)
    scenarios = load_scenarios()
    replies = fixture_replies(scenarios)
    replies[sorted(scenarios)[0]] = Reply(0, ODD_STDERR)
    host.fake.replies.update(replies)
    out = tmp_path / "out"
    started = datetime.now(timezone.utc)
    run = run_tool(tool, ["--out", str(out)], capsys)
    finished = datetime.now(timezone.utc)
    assert run.status == 0, run.out + run.err
    return Capture(out, run, read_manifest(out), scenarios, replies, started, finished)


# ---------------------------------------------------------------------------
# Refusals


@pytest.mark.parametrize("give_out", [False, True], ids=["default-out", "explicit-out"])
@pytest.mark.parametrize("missing", ["TMPDIR", "LASSI_RUNS_ROOT"])
def test_the_tool_refuses_without_a_scratch_variable(
    tool: ModuleType,
    host: Host,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
    give_out: bool,
) -> None:
    # Without TMPDIR a compiler writes to /tmp, and without LASSI_RUNS_ROOT the captures have no scratch home;
    # on the build host /tmp is the root filesystem (Agent Rule 7).
    monkeypatch.delenv(missing)
    out = tmp_path / "out"
    run = run_tool(tool, ["--out", str(out)] if give_out else [], capsys)
    assert run.status != 0
    assert missing in run.out + run.err
    assert host.fake.calls == [], "nothing may run before the refusal"
    assert not out.exists()
    assert not (host.runs_root / "fixture-captures").exists()


@pytest.mark.parametrize("fault", ["banner-without-the-pin", "version-exits-nonzero"])
@pytest.mark.parametrize("toolchain", sorted(PRESETS))
def test_the_tool_refuses_a_compiler_that_is_not_the_pinned_version(
    tool: ModuleType,
    host: Host,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    toolchain: str,
    fault: str,
) -> None:
    # The manifest labels the capture with the pin's VERSION, so the executable must be that version (Agent Rule 10).
    host.fake.replies.update(fixture_replies(load_scenarios()))
    if fault == "banner-without-the-pin":
        host.fake.banners[toolchain] = VERSION_BANNER
    else:
        host.fake.version_status[toolchain] = 1
    out = tmp_path / "out"
    run = run_tool(tool, ["--out", str(out)], capsys)
    assert run.status != 0
    assert "EXPECT_VERSION" in run.out + run.err or "--version" in run.out + run.err
    assert host.fake.compiles() == [], "no scenario may compile with an unverified compiler"
    assert not out.exists(), "nothing is created before every toolchain checks out"


@pytest.mark.parametrize("where", ["inside-the-repository", "outside-lassi-scratch", "non-empty-dir", "a-file"])
def test_the_tool_refuses_a_bad_out_dir(
    tool: ModuleType,
    host: Host,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    where: str,
) -> None:
    # Captures stay out of git and on the scratch disk (Agent Rule 7), and never mix with an earlier capture.
    host.fake.replies.update(fixture_replies(load_scenarios()))
    monkeypatch.setenv("LASSI_SCRATCH", str(tmp_path))
    out = {
        "inside-the-repository": REPO / "capture-refused-by-test",
        "outside-lassi-scratch": tmp_path.parent / (tmp_path.name + "-elsewhere") / "out",
        "non-empty-dir": tmp_path / "used",
        "a-file": tmp_path / "file",
    }[where]
    if where == "non-empty-dir":
        out.mkdir()
        (out / "earlier.stderr").write_bytes(b"")
    elif where == "a-file":
        out.write_bytes(b"")
    run = run_tool(tool, ["--out", str(out)], capsys)
    assert run.status != 0
    assert "out dir" in run.out + run.err
    assert host.fake.calls == [], "nothing may run before the refusal"
    if where in ("inside-the-repository", "outside-lassi-scratch"):
        assert not out.exists()


def test_an_out_dir_inside_lassi_scratch_is_accepted(
    tool: ModuleType,
    host: Host,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenarios = load_scenarios()
    chosen = sorted(scenarios)[0]
    host.fake.replies.update(fixture_replies(scenarios))
    monkeypatch.setenv("LASSI_SCRATCH", str(tmp_path))
    run = run_tool(tool, ["--out", str(tmp_path / "out"), "--only", chosen], capsys)
    assert run.status == 0, run.out + run.err
    assert (tmp_path / "out" / f"{chosen}.stderr").is_file()


# ---------------------------------------------------------------------------
# One full capture


def test_capture_copies_each_stderr_byte_for_byte(capture: Capture) -> None:
    assert sorted(path.stem for path in capture.out.glob("*.stderr")) == sorted(capture.scenarios)
    for name, reply in capture.replies.items():
        captured = (capture.out / f"{name}.stderr").read_bytes()
        assert captured == reply.stderr, name
        assert captured == (capture.out / "work" / name / "compile.stderr").read_bytes(), name


def test_capture_builds_each_scenario_once_in_a_fresh_workdir_holding_its_source_tree(
    capture: Capture, host: Host
) -> None:
    compiles = host.fake.compiles()
    assert sorted(call.cwd.name for call in compiles if call.cwd is not None) == sorted(capture.scenarios)
    assert len(compiles) == len(capture.scenarios)
    for call in compiles:
        assert call.cwd is not None
        name = call.cwd.name
        assert call.cwd.resolve() == (capture.out / "work" / name).resolve()
        workdir = capture.out / "work" / name
        written = {
            path.relative_to(workdir).as_posix(): path.read_bytes()
            for path in workdir.rglob("*")
            if path.is_file() and path.relative_to(workdir).as_posix() not in RESERVED
        }
        assert written == source_bytes(name), f"{name}: the workdir must hold exactly the scenario's sources"


def test_capture_builds_each_toolchain_as_the_stage_runner_does(capture: Capture, host: Host) -> None:
    for call in host.fake.compiles():
        assert call.cwd is not None
        entry = capture.scenarios[call.cwd.name]
        toolchain = entry["toolchain"]
        assert Path(call.argv[0]) == host.root / BINARIES[toolchain], call.cwd.name
        assert call.env == host.compile_env[toolchain], f"{call.cwd.name}: the compile environment is not clean"
    versions = host.fake.versions()
    assert versions, "the tool records each toolchain's --version"
    for call in versions:
        toolchain = next(name for name, binary in BINARIES.items() if Path(call.argv[0]) == host.root / binary)
        assert call.env == host.compile_env[toolchain], "--version runs with the compile environment"


def test_capture_runs_the_adapter_command_with_only_the_override_changed(capture: Capture, host: Host) -> None:
    overridden = [name for name, entry in capture.scenarios.items() if set(entry) & {"arch", "gpu"}]
    assert overridden, "the driver fatal needs an unsupported arch, so at least one scenario overrides its preset"
    for call in host.fake.compiles():
        assert call.cwd is not None
        name = call.cwd.name
        entry = capture.scenarios[name]
        expected = expected_argv(entry, name, str(host.root / BINARIES[entry["toolchain"]]))
        assert call.argv[1:] == expected[1:], name
    for name in overridden:
        entry = capture.scenarios[name]
        key, _ = OVERRIDES[entry["toolchain"]]
        flag = {"arch": f"-arch={entry[key]}", "gpu": f"-gpu={entry[key]}"}[key]
        assert flag in capture.manifest["scenarios"][name]["argv"], name


def test_manifest_records_where_and_when_the_capture_ran(capture: Capture) -> None:
    manifest = capture.manifest
    assert MANIFEST_KEYS <= set(manifest), f"missing {sorted(MANIFEST_KEYS - set(manifest))}"
    date = datetime.fromisoformat(manifest["date"])
    assert date.tzinfo is not None and date.utcoffset() is not None, "the date carries its offset"
    assert capture.started - timedelta(seconds=2) <= date <= capture.finished + timedelta(seconds=2)
    assert manifest["commit"] == git_output("rev-parse", "HEAD").strip()
    assert manifest["dirty"] is bool(git_output("status", "--porcelain").strip())
    subject = git_output("log", "-1", "--format=%s", "HEAD").strip()
    snapshot = git_output("rev-parse", "HEAD^").strip() if subject.startswith(SNAPSHOT_PREFIX) else None
    assert manifest["snapshot_of"] == snapshot, "an rx snapshot HEAD names the commit it was made from"
    assert manifest["rx_run_id"] == RX_RUN_ID
    assert manifest["host"] == platform.node()


def test_manifest_records_each_toolchain(capture: Capture, host: Host) -> None:
    used = {entry["toolchain"] for entry in capture.scenarios.values()}
    toolchains = capture.manifest["toolchains"]
    assert set(toolchains) == used
    for name, entry in toolchains.items():
        assert TOOLCHAIN_KEYS <= set(entry), f"{name}: missing {sorted(TOOLCHAIN_KEYS - set(entry))}"
        assert entry["pins"] == PIN_VERSIONS[name]
        assert entry["pins"] == {pin: read_pin(pin)["VERSION"] for pin in PIN_VERSIONS[name]}
        assert Path(entry["executable"]) == host.root / BINARIES[name]
        assert sorted(entry["environment"]) == sorted(host.compile_env[name]), name
        assert entry["locale"] == LOCALE
        assert entry["version"] == [pinned_line(name), *VERSION_LINES], f"{name}: the --version lines, in order"
        assert entry["version_exit_status"] == 0
        assert entry["pin_files"] == {pin: read_pin(pin) for pin in PIN_VERSIONS[name]}


def test_manifest_records_each_scenario(capture: Capture, host: Host) -> None:
    scenarios = capture.manifest["scenarios"]
    assert set(scenarios) == set(capture.scenarios)
    ran = {call.cwd.name: call.argv for call in host.fake.compiles() if call.cwd is not None}
    for name, entry in scenarios.items():
        assert SCENARIO_KEYS <= set(entry), f"{name}: missing {sorted(SCENARIO_KEYS - set(entry))}"
        reply = capture.replies[name]
        assert entry["toolchain"] == capture.scenarios[name]["toolchain"]
        assert entry["overrides"] == expected_overrides(capture.scenarios[name]), f"{name}: the override it built"
        assert entry["argv"] == ran[name], f"{name}: the argv the adapter ran"
        for argument in entry["argv"][1:]:
            assert not Path(argument).is_absolute() and not PurePosixPath(argument).is_absolute(), (name, argument)
        assert entry["exit_status"] == reply.status, name
        assert entry["stderr_sha256"] == hashlib.sha256(reply.stderr).hexdigest(), name
        assert entry["stderr_bytes"] == len(reply.stderr), name
        assert entry["diagnostics"] == parsed_count(capture.scenarios[name], name, reply.stderr), name


def test_manifest_shows_no_environment_value_except_the_locale(capture: Capture) -> None:
    texts = strings_in(capture.manifest)
    for marker in (HOME_MARKER, TMPDIR_MARKER, PATH_MARKER):
        assert not [text for text in texts if marker in text], f"a value holding {marker} reached the manifest"
    assert not [text for text in texts if "LEAKED" in text], "a variable outside the compile environment leaked"


def test_capture_prints_the_out_dir_and_one_line_per_scenario(capture: Capture) -> None:
    printed = capture.run.out
    assert str(capture.out) in printed or capture.out.as_posix() in printed
    lines = printed.splitlines()
    for name in capture.scenarios:
        assert any(name in line for line in lines), f"no line for {name}"


# ---------------------------------------------------------------------------
# --only and the default out dir


def test_only_captures_the_named_scenarios(
    tool: ModuleType, host: Host, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scenarios = load_scenarios()
    chosen = [min(name for name, entry in scenarios.items() if entry["toolchain"] == preset) for preset in PRESETS]
    host.fake.replies.update(fixture_replies(scenarios))
    out = tmp_path / "out"
    run = run_tool(tool, ["--out", str(out), "--only", *chosen], capsys)
    assert run.status == 0, run.out + run.err
    assert sorted(path.stem for path in out.glob("*.stderr")) == sorted(chosen)
    manifest = read_manifest(out)
    assert sorted(manifest["scenarios"]) == sorted(chosen)
    assert sorted(manifest["toolchains"]) == sorted(PRESETS)
    assert sorted(call.cwd.name for call in host.fake.compiles() if call.cwd is not None) == sorted(chosen)


def test_only_one_toolchain_runs_no_other_compiler(
    tool: ModuleType, host: Host, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scenarios = load_scenarios()
    chosen = min(name for name, entry in scenarios.items() if entry["toolchain"] == "nvcc-sm80")
    host.fake.replies.update(fixture_replies(scenarios))
    out = tmp_path / "out"
    run = run_tool(tool, ["--out", str(out), "--only", chosen], capsys)
    assert run.status == 0, run.out + run.err
    assert list(read_manifest(out)["toolchains"]) == ["nvcc-sm80"]
    started = {Path(call.argv[0]) for call in host.fake.calls if host.fake.pinned(call.argv[0])}
    assert started == {host.root / NVCC_BIN}


def test_default_out_is_named_by_the_rx_run_id_under_lassi_runs_root(
    tool: ModuleType,
    host: Host,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LASSI_RX_RUN_ID", RX_RUN_ID)
    scenarios = load_scenarios()
    chosen = sorted(scenarios)[0]
    host.fake.replies.update(fixture_replies(scenarios))
    run = run_tool(tool, ["--only", chosen], capsys)
    assert run.status == 0, run.out + run.err
    out = host.runs_root / "fixture-captures" / RX_RUN_ID
    assert (out / f"{chosen}.stderr").is_file()
    assert read_manifest(out)["rx_run_id"] == RX_RUN_ID
    assert str(out) in run.out or out.as_posix() in run.out


def test_default_out_without_an_rx_run_id_is_a_utc_timestamp(
    tool: ModuleType, host: Host, capsys: pytest.CaptureFixture[str]
) -> None:
    scenarios = load_scenarios()
    chosen = sorted(scenarios)[0]
    host.fake.replies.update(fixture_replies(scenarios))
    run = run_tool(tool, ["--only", chosen], capsys)
    assert run.status == 0, run.out + run.err
    (out,) = list((host.runs_root / "fixture-captures").iterdir())
    assert re.fullmatch(r"[0-9]{8}[T-]?[0-9]{6}Z?", out.name), out.name
    assert (out / f"{chosen}.stderr").is_file()
    assert read_manifest(out)["rx_run_id"] is None


# ---------------------------------------------------------------------------
# An rx snapshot HEAD, and overrides the tool refuses


def throwaway_git(repo: Path, *args: str) -> str:
    """Run git in the throwaway `repo` with the test identity from the environment; return its stdout, stripped."""
    env = {**os.environ, **IDENTITY}
    done = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, env=env, timeout=60, check=True
    )
    return done.stdout.strip()


def test_snapshot_of_names_the_commit_an_rx_snapshot_was_made_from(
    tool: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # rx sends a dirty tree as a snapshot commit, so the slot's tree is clean and dirty alone reads as reportable.
    repo = tmp_path / "repo"
    repo.mkdir()
    throwaway_git(repo, "init", "--quiet", "-b", "main")
    tree = throwaway_git(repo, "write-tree")
    base = throwaway_git(repo, "commit-tree", tree, "-m", "P0.99: a base commit")
    snapshot = throwaway_git(repo, "commit-tree", tree, "-p", base, "-m", f"{SNAPSHOT_PREFIX}{base[:12]}")
    later = throwaway_git(repo, "commit-tree", tree, "-p", base, "-m", "P0.99: a commit on top")
    monkeypatch.setattr(tool, "REPO", repo)
    for head, expected in ((base, None), (later, None), (snapshot, base)):
        throwaway_git(repo, "update-ref", "--no-deref", "HEAD", head)
        assert tool.snapshot_of() == expected, head


@pytest.mark.parametrize(
    ("override", "reason", "other_reason"),
    [
        ({"gpu": "cc70"}, "nvcc-sm80 has no GPU class attribute", "must be a string"),
        ({"arch": 35}, "'arch' must be a string", "has no"),
    ],
    ids=["no-such-attribute", "not-a-string"],
)
def test_a_bad_override_is_refused_with_its_own_reason(
    tool: ModuleType, tmp_path: Path, override: dict[str, Any], reason: str, other_reason: str
) -> None:
    path = tmp_path / "scenarios.json"
    entry = {"toolchain": "nvcc-sm80", "scenario": "a bad override", **override}
    path.write_text(json.dumps({"nvcc_bad_override": entry}), encoding="ascii")
    (tmp_path / "sources" / "nvcc_bad_override").mkdir(parents=True)
    with pytest.raises(tool.CaptureError) as raised:
        tool.load_scenarios(path, tmp_path / "sources")
    assert reason in str(raised.value) and other_reason not in str(raised.value), str(raised.value)


# ---------------------------------------------------------------------------
# The tool's source


def imported_roots(tree: ast.Module) -> set[str]:
    """Return the top-level package of every absolute import in a module."""
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def is_typed(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return True when every parameter except self or cls, and the return value, are annotated."""
    args = node.args
    params = [*args.posonlyargs, *args.args, *args.kwonlyargs]
    params += [arg for arg in (args.vararg, args.kwarg) if arg is not None]
    params = [param for param in params if param.arg not in ("self", "cls")]
    return node.returns is not None and all(param.annotation is not None for param in params)


def test_the_tool_is_documented_typed_ascii_and_builds_through_the_runner() -> None:
    raw = TOOL.read_bytes()
    assert raw.isascii(), f"{TOOL} is not plain ASCII"
    tree = ast.parse(raw.decode("ascii"))
    assert ast.get_docstring(tree), "the tool has no module docstring"
    functions = [node for node in ast.walk(tree) if isinstance(node, FUNCTION_NODES)]
    public = [node for node in tree.body if isinstance(node, FUNCTION_NODES) and not node.name.startswith("_")]
    assert any(node.name == "main" for node in public), "the tool has no main()"
    assert not [node.name for node in public if not ast.get_docstring(node)], "public functions need docstrings"
    assert not [node.name for node in functions if not is_typed(node)], "every function needs type hints"
    referenced = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    referenced |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    referenced |= {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "build_toolchain" in referenced, "the tool builds each toolchain with lassi.core.runner.build_toolchain"
    assert not DUPLICATED_LOGIC & referenced, f"the tool repeats the runner's logic: {DUPLICATED_LOGIC & referenced}"
    third_party = imported_roots(tree) - set(sys.stdlib_module_names) - {"lassi"}
    assert third_party <= ALLOWED_THIRD_PARTY, f"the tool imports {sorted(third_party)}"


# ---------------------------------------------------------------------------
# On the build host, with the real pinned compilers


def pinned_problem() -> str:
    """Return why this host cannot run the pinned compilers, or "" when it can."""
    if not sys.platform.startswith("linux"):
        return f"the pinned compilers run on the Linux build host, not {sys.platform}"
    for name in ("LASSI_TOOLCHAINS", "LASSI_RUNS_ROOT", "TMPDIR"):
        if not os.environ.get(name):
            return f"{name} is not set"
    root = Path(os.environ["LASSI_TOOLCHAINS"])
    missing = [str(root / binary) for binary in BINARIES.values() if not (root / binary).is_file()]
    return f"no pinned compiler at {', '.join(missing)}" if missing else ""


PINNED_PROBLEM = pinned_problem()


@pytest.mark.remote
@pytest.mark.slow
@pytest.mark.skipif(bool(PINNED_PROBLEM), reason=PINNED_PROBLEM or "the pinned compilers are installed")
def test_capture_with_the_pinned_compilers(tmp_path: Path) -> None:
    # Warnings and clean builds exit 0; every other scenario is an error of its class and exits nonzero.
    out = tmp_path / "capture"
    done = subprocess.run(
        [sys.executable, str(TOOL), "--out", str(out)], cwd=REPO, capture_output=True, text=True, timeout=3600
    )
    assert done.returncode == 0, done.stdout + done.stderr
    scenarios = load_scenarios()
    manifest = read_manifest(out)
    assert set(manifest["scenarios"]) == set(scenarios)
    for name, entry in manifest["scenarios"].items():
        captured = (out / f"{name}.stderr").read_bytes()
        assert captured == (out / "work" / name / "compile.stderr").read_bytes(), name
        assert entry["stderr_sha256"] == hashlib.sha256(captured).hexdigest(), name
        passes = name.endswith("_clean") or "warning" in name
        assert (entry["exit_status"] == 0) == passes, f"{name} exited {entry['exit_status']}"
        if not passes:
            assert captured, f"{name} failed without printing anything"
    for name, preset in PRESETS.items():
        expect = read_pin(preset.PIN)["EXPECT_VERSION"]
        assert any(expect in line for line in manifest["toolchains"][name]["version"]), name
