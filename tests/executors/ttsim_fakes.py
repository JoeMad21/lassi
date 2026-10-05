"""Shared fakes for the tests of the ttsim executor and its smoke driver (task P4.11); not a test module.

- make_install builds a toolchains root that the ttsim executor's install
  checks accept, without the pinned files: ttsim@<VERSION>/ holds
  PLACEHOLDER bytes as the library and the SoC descriptor, and a copy of
  toolchains/ (the pin files and the tracked list of CMake-fetched
  packages), with ttsim.pin's SHA256, SIZE, and SOC_DESCRIPTOR_SHA256
  rewritten to match those bytes, replaces lassi.toolchains.pins.PINS_DIR.
  The tt-metal tree holds only lassi-install.txt, whose first line names
  the pin's NAME, COMMIT, and URL as toolchains/tt-metal.sh writes it, and
  lassi-cpm-sources.txt, the tracked list without its comments. Nothing
  there is the pinned library, descriptor, or tree.
- FakeSandbox stands in for lassi.executors.sandbox.Sandbox: it records each
  call's SandboxSpec, argv, and Limits, applies that call's effect (the way
  a program writes files), and returns its canned SandboxResults in order.
  Nothing is executed.
- capture(name) returns the run recorded in
  tests/executors/fixtures/ttsim/<name>/ (P4.9, README.md there) as a
  SandboxResult; watcher_log() returns that folder's Watcher log, and
  writes_watcher_log() an effect that puts a log where a Watcher rerun's
  TT_METAL_LOGS_PATH says tt-metal writes it.

The wall times given to canned results are PLACEHOLDER values. No value here
is a measurement.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from lassi.core.interfaces import Limits
from lassi.executors import sandbox as sandbox_module
from lassi.toolchains import pins as pins_module

REPO = Path(__file__).resolve().parents[2]
TOOLCHAINS_DIR = REPO / "toolchains"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "ttsim"
# The six P4.9 captures (README.md in FIXTURES).
CAPTURES = (
    "non-contractual-behavior",
    "undefined-behavior",
    "unsupported-functionality",
    "jit-error",
    "hang",
    "watcher-hang",
)
# The working directory of the P4.9 step seed-jit-error, the prefix of the kernel path its compiler lines print.
P49_CWD = (
    "/mnt/nvme10/joseph_ufl/lassi-runs/p49-ttsim-runtime/20260926-140947-p49-ttsim-runtime-6f4c/raw/seed-jit-error/cwd"
)
# The seeded kernel's path, as the program named it in the P4.9 runs (jit-error and watcher-hang print it).
SEEDED_KERNEL = "tt_metal/programming_examples/add_2_integers_in_riscv/kernels/reader_writer_add_in_riscv.cpp"
# PLACEHOLDER bytes standing in for the pinned library and SoC descriptor.
LIBRARY_BYTES = b"PLACEHOLDER: stands in for the pinned ttsim library in tests; not a library\n"
DESCRIPTOR_BYTES = b"# PLACEHOLDER: stands in for the pinned SoC descriptor in tests\n"
# The directory name inside a workdir that the executor reserves for itself (no model path may start with "@").
RESERVED = "@ttsim"
TTSIM_PIN, TT_METAL_PIN = "ttsim", "tt-metal"
TRACKED_CPM_SOURCES = "tt-metal-cpm-sources.txt"
PLACEHOLDER_WALL_S = 0.25
# The environment names the ttsim executor sets, and the one its Watcher rerun adds (the bible's ttsim row).
TTSIM_NAMES = (
    "TT_METAL_SIMULATOR",
    "TT_METAL_SLOW_DISPATCH_MODE",
    "TT_METAL_DISABLE_SFPLOADMACRO",
    "TT_METAL_RUNTIME_ROOT",
    "TT_METAL_CACHE",
    "TT_METAL_LOGS_PATH",
    "TT_METAL_INSPECTOR_RPC",
    "TT_METAL_THREADCOUNT",
)
WATCHER_NAME = "TT_METAL_WATCHER"


def sha256(data: bytes) -> str:
    """Return the hex sha256 of `data`."""
    return hashlib.sha256(data).hexdigest()


def same_path(first: object, second: object) -> bool:
    """Return True when two paths name the same place once symbolic links and separators are resolved."""
    return os.path.realpath(str(first)) == os.path.realpath(str(second))


def inside(path: object, root: object) -> bool:
    """Return True when `path` lies strictly inside `root`, both resolved."""
    resolved, base = Path(os.path.realpath(str(path))), Path(os.path.realpath(str(root)))
    return base in resolved.parents


def path_forms(path: Path) -> set[str]:
    """Return the spellings a message may use for `path`: as given or resolved, native or POSIX."""
    resolved = Path(os.path.realpath(path))
    return {str(path), path.as_posix(), str(resolved), resolved.as_posix()}


def names_path(message: str, path: Path) -> bool:
    """Return True when `message` holds one spelling of `path`."""
    return any(form in message for form in path_forms(path))


def _rewrite_pin(text: str, values: Mapping[str, str]) -> str:
    """Return pin file text with the KEY=value line of each key in `values` replaced (a trailing comment dropped)."""
    lines = []
    for line in text.splitlines():
        key = line.split("=", 1)[0].strip()
        replace = "=" in line and not line.lstrip().startswith("#") and key in values
        lines.append(f"{key}={values[key]}" if replace else line)
    return "".join(f"{line}\n" for line in lines)


@dataclass(frozen=True)
class FakeInstall:
    """A toolchains root the install checks accept, and where each checked file lies."""

    root: Path
    pins_dir: Path
    library: Path
    descriptor: Path
    tree: Path

    @property
    def record(self) -> Path:
        """Return the tree's install record, lassi-install.txt."""
        return self.tree / "lassi-install.txt"


def make_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ttsim_values: Mapping[str, str] | None = None
) -> FakeInstall:
    """Build the PLACEHOLDER install under tmp_path and point the pin reader at a patched copy of toolchains/.

    `ttsim_values` replaces more KEY=value lines of the copied ttsim.pin
    (for example VERSION), after SHA256, SIZE, and SOC_DESCRIPTOR_SHA256.
    """
    pins_dir = tmp_path / "pins"
    pins_dir.mkdir()
    for path in TOOLCHAINS_DIR.iterdir():
        if path.is_file():
            shutil.copyfile(path, pins_dir / path.name)
    values = {
        "SHA256": sha256(LIBRARY_BYTES),
        "SIZE": str(len(LIBRARY_BYTES)),
        "SOC_DESCRIPTOR_SHA256": sha256(DESCRIPTOR_BYTES),
        **(ttsim_values or {}),
    }
    pin_file = pins_dir / f"{TTSIM_PIN}.pin"
    pin_file.write_bytes(_rewrite_pin(pin_file.read_bytes().decode("utf-8"), values).encode("utf-8"))
    monkeypatch.setattr(pins_module, "PINS_DIR", pins_dir)
    ttsim, metal = pins_module.read_pin(TTSIM_PIN), pins_module.read_pin(TT_METAL_PIN)
    root = tmp_path / "toolchains"
    prefix = root / ttsim["PREFIX_NAME"]
    prefix.mkdir(parents=True)
    library, descriptor = prefix / ttsim["LIBRARY"], prefix / ttsim["SOC_DESCRIPTOR"]
    library.write_bytes(LIBRARY_BYTES)
    descriptor.write_bytes(DESCRIPTOR_BYTES)
    tree = root / metal["PREFIX_NAME"]
    tree.mkdir()
    (tree / "lassi-install.txt").write_bytes(f"{metal['NAME']} {metal['COMMIT']} {metal['URL']}\n".encode("ascii"))
    tracked = (pins_dir / TRACKED_CPM_SOURCES).read_bytes().decode("utf-8").splitlines()
    entries = [line.rstrip() for line in tracked if line.strip() and not line.startswith("#")]
    (tree / "lassi-cpm-sources.txt").write_bytes("".join(f"{line}\n" for line in entries).encode("utf-8"))
    return FakeInstall(root=root, pins_dir=pins_dir, library=library, descriptor=descriptor, tree=tree)


def make_artifact(runs_root: Path, name: str = "main") -> Path:
    """Return a placeholder artifact in a fresh build directory under the runs root, with a kernel beside it.

    The workdir's pre-run files are the artifact and kernels/k.cpp; neither
    is ever executed or compiled.
    """
    workdir = runs_root / "attempt00" / "build"
    (workdir / "kernels").mkdir(parents=True)
    artifact = workdir / name
    artifact.write_bytes(b"PLACEHOLDER artifact, never executed\n")
    (workdir / "kernels" / "k.cpp").write_bytes(b"// SYNTHETIC kernel placed beside the artifact; never compiled\n")
    return artifact


def canned(**overrides: Any) -> Any:
    """Return a SandboxResult for a normal exit with status 0 and no output, with any field overridden."""
    values: dict[str, Any] = {"returncode": 0, "stdout": "", "stderr": "", "wall_s": PLACEHOLDER_WALL_S}
    values.update({"hang": False, "killed": False})
    values.update(overrides)
    return sandbox_module.SandboxResult(**values)


def capture_text(name: str, file: str) -> str:
    """Return a capture's file as text, decoded as UTF-8 with its newlines kept."""
    return (FIXTURES / name / file).read_bytes().decode("utf-8")


def capture_status(name: str) -> dict[str, Any]:
    """Return a capture's status.json."""
    data = json.loads(capture_text(name, "status.json"))
    assert isinstance(data, dict)
    return data


def capture(name: str) -> Any:
    """Return the capture `name` as the SandboxResult a sandboxed run of it would give.

    The exit status is the capture's; a run stopped at its limit (timed_out)
    reads as a hang, as the sandbox classifies a run killed at its wall limit.
    The wall time is a PLACEHOLDER.
    """
    status = capture_status(name)
    return canned(
        returncode=status["rc"],
        stdout=capture_text(name, "stdout.txt"),
        stderr=capture_text(name, "stderr.txt"),
        hang=bool(status["timed_out"]),
    )


def watcher_log() -> str:
    """Return the watcher-hang capture's Watcher log."""
    return capture_text("watcher-hang", "watcher.log")


Effect = Callable[[Any, list[str]], None]


def writes_watcher_log(text: str) -> Effect:
    """Return an effect that writes `text` where tt-metal writes Watcher's log under the call's TT_METAL_LOGS_PATH."""

    def effect(spec: Any, argv: list[str]) -> None:
        """Write the log for a run with Watcher on; a run without it fails the test."""
        environment = dict(spec.environment or {})
        assert environment.get(WATCHER_NAME) == "1", "only a run with TT_METAL_WATCHER=1 writes a Watcher log"
        path = Path(environment["TT_METAL_LOGS_PATH"]) / "generated" / "watcher" / "watcher.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))

    return effect


@dataclass(frozen=True)
class SandboxCall:
    """One call a FakeSandbox received."""

    spec: Any
    argv: list[str]
    limits: Limits

    @property
    def environment(self) -> dict[str, str]:
        """Return a copy of the program environment the call's spec sets ({} for none)."""
        return dict(self.spec.environment or {})


@dataclass
class FakeSandbox:
    """A stand-in for lassi.executors.sandbox.Sandbox that returns `results` in order; nothing is executed.

    `effects[n]`, when set, runs during call n with the spec and argv, the
    way a program writes files into its workdir. A call past `results`
    fails the test.
    """

    results: Sequence[Any]
    effects: Sequence[Effect | None] = ()
    calls: list[SandboxCall] = field(default_factory=list)

    def run(self, spec: Any, argv: Sequence[str], limits: Limits) -> Any:
        """Record the call, apply its effect, and return its canned result."""
        index = len(self.calls)
        self.calls.append(SandboxCall(spec=spec, argv=list(argv), limits=limits))
        assert index < len(self.results), f"the executor started run {index + 1}; the test expects {len(self.results)}"
        if index < len(self.effects) and self.effects[index] is not None:
            effect = self.effects[index]
            assert effect is not None
            effect(spec, list(argv))
        return self.results[index]
