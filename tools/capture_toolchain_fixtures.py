"""Capture the toolchain stderr fixtures with the pinned compilers (task P0.15; bible Component Interfaces, Toolchain).

Usage (on the build host, through tools/rx.py):

    uv run python tools/capture_toolchain_fixtures.py [--out DIR] [--only NAME ...] [--show]

tests/toolchains/fixtures/scenarios.json names each scenario: its toolchain
(a registry name), an optional override of the preset's ARCH ("arch") or GPU
("gpu") class attribute, and a one-line description. The files a scenario
compiles are tests/toolchains/fixtures/sources/<scenario>/, as relative
paths. These sources are hand-written fixtures; the tool only compiles them
and never runs a built program.

Each toolchain is built exactly as the stage runner builds it, with
lassi.core.runner.build_toolchain: the pinned executable under
$LASSI_TOOLCHAINS, the clean compile environment (PATH, LANG=C, LC_ALL=C,
HOME, TMPDIR, and linked prefixes such as NVHPC_CUDA_HOME), and EnvRunner. An
override builds a subclass of the preset that sets only that attribute, so
the command line keeps the adapter's form. Each scenario builds once, with
the adapter's own build(), in a fresh workdir <out>/work/<scenario>/, and
its compile.stderr is copied byte for byte to <out>/<scenario>.stderr.

<out>/manifest.json (plain ASCII JSON) records the capture's provenance
(Agent Rules 1 and 10): date (ISO 8601 with an offset), commit and dirty
(from git), snapshot_of, rx_run_id ($LASSI_RX_RUN_ID or null), host, and
per toolchain its pin versions, every pair of each pin file, executable,
environment variable names (values never, except the locale), and the
non-blank lines and exit status of `<executable> --version` run in the
same environment; per scenario its
overrides (class attribute -> value, empty for the preset), the argv the
adapter ran (the sources relative to the workdir), the exit status, the
stderr sha256 and size, and how many Diagnostics the adapter parses from it.

tools/rx.py sends a dirty working tree as a snapshot commit, which is clean
inside the slot, so dirty is false there; snapshot_of then names the commit
the snapshot was made from. A capture is reportable only when dirty is false
and snapshot_of is null (AGENTS.md, Results). Before anything is created,
each toolchain's --version must exit 0 and print its pin's EXPECT_VERSION,
so a capture is never labeled with a pin its compiler is not (Agent Rule
10). The copy is byte for byte to
compile.stderr, which build() writes from the compiler's stderr decoded as
UTF-8 with errors replaced: a byte that is not valid UTF-8 arrives as
U+FFFD, exactly as a run records it.

The default out dir is $LASSI_RUNS_ROOT/fixture-captures/<$LASSI_RX_RUN_ID,
else the UTC time as YYYYMMDD-HHMMSS>. The tool refuses to run without
TMPDIR or LASSI_RUNS_ROOT, since a compiler would otherwise write to /tmp on
the root filesystem (Agent Rule 7); an out dir must be new or empty, outside
the repository, and inside $LASSI_SCRATCH when that is set. It prints the
out dir and one line per scenario; --show also prints each captured stderr,
with any byte that is not ASCII shown as a backslash escape.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from lassi.core.registry import DEFAULT_REGISTRY, Registry  # noqa: E402
from lassi.core.runner import BuiltToolchain, RunError, build_toolchain  # noqa: E402
from lassi.toolchains import STDERR_ATTACHMENT, CommandResult, CommandRunner  # noqa: E402

FIXTURES = REPO / "tests" / "toolchains" / "fixtures"
SCENARIOS_JSON = FIXTURES / "scenarios.json"
SOURCES = FIXTURES / "sources"

# Where the default out dir goes under $LASSI_RUNS_ROOT, and the names inside an out dir.
CAPTURES_DIR = "fixture-captures"
WORK_DIR = "work"
MANIFEST = "manifest.json"

# The scenarios.json keys that override a preset class attribute, and that attribute.
OVERRIDES = {"arch": "ARCH", "gpu": "GPU"}
ENTRY_KEYS = frozenset({"toolchain", "scenario", *OVERRIDES})
# The only compile environment values the manifest shows.
LOCALE_NAMES = ("LANG", "LC_ALL")
# The subject tools/rx.py gives the snapshot commit of a dirty working tree; the group is the commit it came from.
SNAPSHOT_SUBJECT = re.compile(r"rx snapshot of ([0-9a-f]{7,40})")
# How many non-blank --version lines the manifest keeps, and how long --version and git may take, in seconds.
VERSION_LINES_MAX = 20
VERSION_TIMEOUT_S = 120.0
GIT_TIMEOUT_S = 60.0

# The status main() returns when it refuses to run or a capture fails.
REFUSED = 2


class CaptureError(RuntimeError):
    """A capture that cannot start or cannot finish; the message says why."""


@dataclass(frozen=True)
class Scenario:
    """One scenario: its name, toolchain registry name, class attribute overrides, and files (path -> text)."""

    name: str
    toolchain: str
    overrides: Mapping[str, str]
    files: Mapping[str, str]


class _Recorder:
    """A CommandRunner that runs each command with the toolchain's own runner and keeps its argv and status."""

    def __init__(self, inner: CommandRunner) -> None:
        """Wrap `inner`, the runner build_toolchain gave the toolchain."""
        self.inner = inner
        self.argv: list[str] | None = None
        self.returncode: int | None = None

    def __call__(self, argv: Sequence[str], cwd: Path, timeout_s: float) -> CommandResult:
        """Run `argv` with the wrapped runner and keep the command and its exit status."""
        result = self.inner(argv, cwd, timeout_s)
        self.argv, self.returncode = list(argv), result.returncode
        return result


# ---------------------------------------------------------------------------
# Scenarios


def read_sources(tree: Path) -> dict[str, str]:
    """Return every file under `tree` as relative POSIX path -> UTF-8 text, in sorted order."""
    if not tree.is_dir():
        raise CaptureError(f"no source tree {tree}")
    return {
        path.relative_to(tree).as_posix(): path.read_bytes().decode("utf-8")
        for path in sorted(tree.rglob("*"))
        if path.is_file()
    }


def _scenario(name: str, entry: Any, sources: Path, path: Path) -> Scenario:
    """Return one checked entry of the scenario manifest `path` with its source files."""
    if not isinstance(entry, dict) or not set(entry) <= ENTRY_KEYS:
        raise CaptureError(f"{path}: {name}: an entry is an object with keys from {sorted(ENTRY_KEYS)}")
    toolchain = entry.get("toolchain")
    if toolchain not in DEFAULT_REGISTRY.names("Toolchain"):
        raise CaptureError(f"{path}: {name}: toolchain {toolchain!r} is not a registered Toolchain")
    preset = DEFAULT_REGISTRY.get("Toolchain", toolchain).factory
    overrides: dict[str, str] = {}
    for key, attribute in OVERRIDES.items():
        if key not in entry:
            continue
        value = entry[key]
        if not hasattr(preset, attribute):
            raise CaptureError(f"{path}: {name}: {toolchain} has no {attribute} class attribute for {key!r}")
        if not isinstance(value, str):
            raise CaptureError(f"{path}: {name}: {key!r} must be a string, not {type(value).__name__}")
        overrides[attribute] = value
    return Scenario(name, toolchain, overrides, read_sources(sources / name))


def load_scenarios(path: Path = SCENARIOS_JSON, sources: Path = SOURCES) -> dict[str, Scenario]:
    """Return the scenarios of scenarios.json with their source trees, by name in sorted order."""
    data = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(data, dict):
        raise CaptureError(f"{path}: expected one JSON object, scenario name -> entry")
    return {name: _scenario(name, data[name], sources, path) for name in sorted(data)}


def scenario_registry(scenario: Scenario) -> Registry:
    """Return the registry that builds the scenario: the default one, or one holding the overriding subclass.

    The subclass of the preset sets only the overridden class attributes and
    is registered under the preset's own name, so build_toolchain pins and
    runs it exactly as it does the preset.
    """
    if not scenario.overrides:
        return DEFAULT_REGISTRY
    preset = DEFAULT_REGISTRY.get("Toolchain", scenario.toolchain).factory
    subclass = type(f"{preset.__name__}Override", (preset,), dict(scenario.overrides))
    registry = Registry()
    registry.register("Toolchain", scenario.toolchain, subclass)
    return registry


# ---------------------------------------------------------------------------
# Capturing


def toolchain_record(built: BuiltToolchain, cwd: Path) -> dict[str, Any]:
    """Return the manifest entry of a pinned toolchain: pins, pin files, executable, variable names, locale, --version.

    Raises CaptureError when the toolchain declares no pin, or when
    `<executable> --version` exits nonzero or does not print the
    EXPECT_VERSION of the toolchain's own pin.
    """
    if built.executable is None or built.environment is None:
        raise CaptureError(f"toolchain {built.name!r} declares no pin; the fixtures come from pinned compilers")
    environment = built.environment
    result = built.toolchain.runner([built.executable, "--version"], cwd, VERSION_TIMEOUT_S)
    output = result.stdout + result.stderr
    pin = type(built.toolchain).PIN
    expected = built.pins[pin].get("EXPECT_VERSION", "")
    if not expected or result.returncode != 0 or expected not in output:
        raise CaptureError(
            f"{built.executable} --version exited {result.returncode} and must print the EXPECT_VERSION of "
            f"toolchains/{pin}.pin ({expected!r}); the capture would not be of the pinned compiler"
        )
    lines = [line.rstrip() for line in output.splitlines() if line.strip()]
    return {
        "pins": {pin: values["VERSION"] for pin, values in built.pins.items()},
        "pin_files": {pin: dict(values) for pin, values in built.pins.items()},
        "executable": built.executable,
        "environment": sorted(environment),
        "locale": {name: environment[name] for name in LOCALE_NAMES if name in environment},
        "version": lines[:VERSION_LINES_MAX],
        "version_exit_status": result.returncode,
    }


def capture_scenario(scenario: Scenario, out: Path, root: Path | None) -> tuple[dict[str, Any], bytes]:
    """Build one scenario in <out>/work/<name>/, copy its stderr to <out>/<name>.stderr; return its entry and bytes."""
    built = build_toolchain(scenario.toolchain, root, scenario_registry(scenario))
    toolchain = built.toolchain
    recorder = _Recorder(toolchain.runner)
    toolchain.runner = recorder
    workdir = out / WORK_DIR / scenario.name
    workdir.mkdir(parents=True)
    toolchain.build(scenario.files, workdir)
    if recorder.argv is None or recorder.returncode is None:
        raise CaptureError(f"{scenario.name}: the adapter ran no compiler; check the scenario's source tree")
    data = (workdir / STDERR_ATTACHMENT).read_bytes()
    (out / f"{scenario.name}.stderr").write_bytes(data)
    diagnostics = toolchain.parse(data.decode("utf-8", errors="replace"), scenario.files)
    entry = {
        "toolchain": scenario.toolchain,
        "overrides": dict(scenario.overrides),
        "argv": recorder.argv,
        "exit_status": recorder.returncode,
        "stderr_sha256": hashlib.sha256(data).hexdigest(),
        "stderr_bytes": len(data),
        "diagnostics": len(diagnostics),
    }
    return entry, data


def _git(*args: str) -> str | None:
    """Return the stdout of `git <args>` run in the repository, or None when git is missing or fails."""
    try:
        done = subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True, timeout=GIT_TIMEOUT_S, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout if done.returncode == 0 else None


def git_state() -> tuple[str | None, bool | None]:
    """Return the repository's commit and whether `git status --porcelain` lists anything; None when git fails."""
    head = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    commit = (head.strip() or None) if head is not None else None
    return commit, (bool(status.strip()) if status is not None else None)


def snapshot_of() -> str | None:
    """Return the commit HEAD was made from when HEAD is an rx snapshot commit, else None.

    tools/rx.py sends a dirty working tree as a snapshot commit with the
    subject 'rx snapshot of <head>' and checks it out in the slot, where the
    tree is then clean: dirty alone cannot tell such an exploratory capture
    from a reportable one. The full parent hash is returned when git can
    resolve it, else the abbreviated one from the subject.
    """
    subject = _git("log", "-1", "--format=%s", "HEAD")
    match = SNAPSHOT_SUBJECT.fullmatch(subject.strip()) if subject is not None else None
    if match is None:
        return None
    parent = (_git("rev-parse", "HEAD^") or "").strip()
    return parent if parent.startswith(match.group(1)) else match.group(1)


def show(name: str, data: bytes) -> None:
    """Print a captured stderr between marker lines, with every byte that is not ASCII as a backslash escape."""
    text = data.decode("ascii", errors="backslashreplace")
    ending = "" if not text or text.endswith("\n") else "\n[no final newline]\n"
    print(f"--- begin {name}.stderr ({len(data)} bytes)\n{text}{ending}--- end {name}.stderr", flush=True)


# ---------------------------------------------------------------------------
# The command line


def _within(path: Path, root: Path) -> bool:
    """Return True when the resolved `path` is the resolved `root` or lies under it."""
    path, root = path.resolve(), root.resolve()
    return path == root or root in path.parents


def out_dir(given: str | None, runs_root: str) -> Path:
    """Return the out dir: `given`, else $LASSI_RUNS_ROOT/fixture-captures/<rx run id or UTC time>.

    Raises CaptureError when it is inside the repository, outside
    $LASSI_SCRATCH when that is set, or exists and is not an empty directory.
    """
    if given is not None:
        out = Path(os.path.abspath(given))
    else:
        name = os.environ.get("LASSI_RX_RUN_ID") or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        out = Path(os.path.abspath(runs_root)) / CAPTURES_DIR / name
    if _within(out, REPO):
        raise CaptureError(f"the out dir {out} is inside the repository {REPO}; captures stay out of git")
    scratch = os.environ.get("LASSI_SCRATCH", "")
    if scratch and not _within(out, Path(scratch)):
        raise CaptureError(f"the out dir {out} is outside $LASSI_SCRATCH ({scratch}) (Agent Rule 7)")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise CaptureError(f"the out dir {out} already exists and is not an empty directory; choose a new one")
    return out


def _arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description="Capture the toolchain stderr fixtures with the pinned compilers.")
    parser.add_argument("--out", help="the out dir (default: $LASSI_RUNS_ROOT/fixture-captures/<rx run id or time>)")
    parser.add_argument("--only", nargs="+", metavar="NAME", help="capture only these scenarios")
    parser.add_argument("--show", action="store_true", help="also print each captured stderr")
    return parser.parse_args(argv)


def _scratch_problem() -> str | None:
    """Return why the tool may not run here, or None when TMPDIR and LASSI_RUNS_ROOT are both set."""
    for name in ("TMPDIR", "LASSI_RUNS_ROOT"):
        if not os.environ.get(name):
            return (
                f"{name} is not set; without it compiler temporary files or captures could land on the root "
                "filesystem (Agent Rule 7). Run this on the build host through tools/rx.py, which sets both"
            )
    return None


def _chosen(scenarios: Mapping[str, Scenario], only: Sequence[str] | None) -> list[Scenario]:
    """Return the scenarios to capture: all of them, or those --only names, in sorted order."""
    if not only:
        return list(scenarios.values())
    unknown = sorted(set(only) - set(scenarios))
    if unknown:
        raise CaptureError(f"no scenario named {', '.join(unknown)}; scenarios: {', '.join(scenarios)}")
    return [scenarios[name] for name in sorted(set(only))]


def capture(chosen: Sequence[Scenario], out: Path, root: Path | None, shown: bool) -> dict[str, Any]:
    """Capture the chosen scenarios into `out` and return the manifest; nothing is created before every build works."""
    date = datetime.now().astimezone().isoformat(timespec="seconds")
    names = sorted({scenario.toolchain for scenario in chosen})
    built = {name: build_toolchain(name, root) for name in names}
    for scenario in chosen:
        build_toolchain(scenario.toolchain, root, scenario_registry(scenario))
    toolchains = {name: toolchain_record(built[name], Path.cwd()) for name in names}
    out.mkdir(parents=True, exist_ok=True)
    print(f"out dir: {out}", flush=True)
    commit, dirty = git_state()
    scenarios: dict[str, Any] = {}
    for scenario in chosen:
        entry, data = capture_scenario(scenario, out, root)
        scenarios[scenario.name] = entry
        print(
            f"{scenario.name}  {scenario.toolchain}  exit {entry['exit_status']}  stderr {entry['stderr_bytes']} "
            f"bytes  diagnostics {entry['diagnostics']}",
            flush=True,
        )
        if shown:
            show(scenario.name, data)
    return {
        "date": date,
        "commit": commit,
        "dirty": dirty,
        "snapshot_of": snapshot_of(),
        "rx_run_id": os.environ.get("LASSI_RX_RUN_ID") or None,
        "host": platform.node(),
        "toolchains": toolchains,
        "scenarios": scenarios,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Capture the scenarios and write the manifest; return 0, or REFUSED after printing why."""
    args = _arguments(argv)
    problem = _scratch_problem()
    if problem is not None:
        print(f"capture_toolchain_fixtures: {problem}", file=sys.stderr)
        return REFUSED
    toolchains = os.environ.get("LASSI_TOOLCHAINS", "")
    try:
        out = out_dir(args.out, os.environ["LASSI_RUNS_ROOT"])
        chosen = _chosen(load_scenarios(), args.only)
        manifest = capture(chosen, out, Path(toolchains) if toolchains else None, args.show)
    except (CaptureError, RunError) as error:
        print(f"capture_toolchain_fixtures: {error}", file=sys.stderr)
        return REFUSED
    (out / MANIFEST).write_bytes((json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("ascii"))
    print(f"manifest: {out / MANIFEST}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
