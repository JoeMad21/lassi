"""Tests for how the runner builds ttkernel-ttmetal with its two pins (task P17.36; Agent Rule 10).

The toolchain declares PIN tt-metal (the host clang++-20 and the installed
tree, as ttmetal-host) and TOOL_PINS {tt-mlir: (TTMLIR_OPT, TTMLIR_TRANSLATE)}.
lassi.core.runner.build_toolchain then also: requires the tt-mlir install
<root>/<PREFIX_NAME> to be a directory inside the root whose
lassi-install.txt names the pin's NAME and COMMIT (check_tool_pin); requires
each tool to be a file resolving inside the root; runs `<tool> --version`
through the compile runner, which must print tt-mlir.pin's EXPECT_VERSION;
records the tt-mlir pin beside the tt-metal pin; and passes the tool paths
as `tools`. Every refusal is a RunError before any build. No tool runs: the
compile runner is a fake that answers --version with a PLACEHOLDER banner.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

import lassi.toolchains  # noqa: F401  (registers every toolchain preset)
from lassi.core import runner as runner_module
from lassi.core.runner import RunError, build_toolchain
from lassi.toolchains import CommandResult
from lassi.toolchains import pins as pins_module

REPO = Path(__file__).resolve().parents[2]
NAME = "ttkernel-ttmetal"
BANNER_LEAD = "PLACEHOLDER banner of a fake compile runner, not a tool"


@pytest.fixture(autouse=True)
def clean_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset the gate variables and give HOME and TMPDIR test directories."""
    for name in ("LASSI_RUNS_ROOT", "LASSI_SCRATCH", "LASSI_TOOLCHAINS"):
        monkeypatch.delenv(name, raising=False)
    for name in ("home", "compile-tmp"):
        (tmp_path / name).mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("TMPDIR", str(tmp_path / "compile-tmp"))


@dataclass
class Log:
    """What the fake compile runner saw, and the banner each executable's --version prints."""

    banners: dict[str, str] = field(default_factory=dict)
    calls: list[list[str]] = field(default_factory=list)


def fake_runner(log: Log) -> type:
    """Return a stand-in for SandboxedCompileRunner that answers --version from the log and runs nothing else."""

    class FakeCompileRunner:
        """Answers --version with the executable's banner; any other command fails the test."""

        def __init__(self, **keywords: Any) -> None:
            self.keywords = keywords

        def spec(self, build: Path) -> None:
            """Accept any layout."""

        def __call__(self, argv: Sequence[str], cwd: Path, timeout_s: float) -> CommandResult:
            log.calls.append(list(argv))
            assert list(argv[1:]) == ["--version"], f"only --version may run here, not {argv!r}"
            return CommandResult(0, log.banners.get(argv[0], "PLACEHOLDER no banner\n"), "")

    return FakeCompileRunner


@dataclass(frozen=True)
class Layout:
    """The stand-in pins, root, trees, and tools."""

    root: Path
    clang: Path
    opt: Path
    translate: Path
    mlir_home: Path
    log: Log


def write_pin(path: Path, pairs: dict[str, str]) -> None:
    """Write a pin file of quoted KEY="value" lines."""
    path.write_bytes(("# SYNTHETIC pin copy\n" + "".join(f'{k}="{v}"\n' for k, v in pairs.items())).encode("ascii"))


@pytest.fixture
def layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Layout:
    """Copy toolchains/, point tt-metal's EXECUTABLE at a stand-in, and build stand-in installs of both pins."""
    pins_dir = tmp_path / "pins"
    pins_dir.mkdir()
    for path in (REPO / "toolchains").iterdir():
        if path.is_file():
            shutil.copyfile(path, pins_dir / path.name)
    metal = pins_module.read_pin("tt-metal")
    mlir = pins_module.read_pin("tt-mlir")
    clang = tmp_path / "host-bin" / "clang++-20"
    clang.parent.mkdir()
    clang.write_bytes(b"")
    write_pin(pins_dir / "tt-metal.pin", {**metal, "EXECUTABLE": str(clang)})
    monkeypatch.setattr(pins_module, "PINS_DIR", pins_dir)
    root = tmp_path / "toolchains-root"
    tree = root / metal["PREFIX_NAME"]
    tree.mkdir(parents=True)
    (tree / "lassi-install.txt").write_bytes(f"tt-metal {metal['COMMIT']} {metal['URL']}\n".encode("ascii"))
    tracked = (REPO / "toolchains" / "tt-metal-cpm-sources.txt").read_text(encoding="utf-8").splitlines()
    cpm = [line for line in tracked if line.strip() and not line.startswith("#")]
    (tree / "lassi-cpm-sources.txt").write_bytes(("\n".join(cpm) + "\n").encode("utf-8"))
    home = root / mlir["PREFIX_NAME"]
    opt, translate = home / mlir["TTMLIR_OPT"], home / mlir["TTMLIR_TRANSLATE"]
    opt.parent.mkdir(parents=True)
    opt.write_bytes(b"")
    translate.write_bytes(b"")
    (home / "lassi-install.txt").write_bytes(f"tt-mlir {mlir['COMMIT']} {mlir['URL']} x\n".encode("ascii"))
    log = Log()
    resolved = root.resolve()
    log.banners[str(clang)] = f"{BANNER_LEAD}\n{metal['EXPECT_VERSION']}\n"
    for tool in (opt, translate):
        log.banners[str(resolved / tool.relative_to(root))] = f"{BANNER_LEAD}\n{mlir['EXPECT_VERSION']}\n"
    monkeypatch.setattr(runner_module, "SandboxedCompileRunner", fake_runner(log))
    return Layout(root, clang, opt, translate, home, log)


def test_build_toolchain_checks_both_pins_and_passes_the_tools(layout: Layout) -> None:
    built = build_toolchain(NAME, layout.root)
    resolved = layout.root.resolve()
    opt = str(resolved / layout.opt.relative_to(layout.root))
    translate = str(resolved / layout.translate.relative_to(layout.root))
    assert layout.log.calls == [[str(layout.clang), "--version"], [opt, "--version"], [translate, "--version"]]
    assert set(built.pins) == {"tt-metal", "tt-mlir"}, "a trial records toolchain_pins.tt_metal and tt_mlir"
    assert built.pins["tt-mlir"] == pins_module.read_pin("tt-mlir")
    assert (built.toolchain.opt, built.toolchain.translate) == (opt, translate)


def test_a_missing_tt_mlir_install_is_refused_before_any_process_runs(layout: Layout) -> None:
    shutil.rmtree(layout.mlir_home)
    with pytest.raises(RunError, match="tt-mlir"):
        build_toolchain(NAME, layout.root)
    assert layout.log.calls == [], "the filesystem checks come before any --version check"


def test_an_unfinished_tt_mlir_install_is_refused(layout: Layout) -> None:
    (layout.mlir_home / "lassi-install.txt").unlink()
    with pytest.raises(RunError, match="lassi-install.txt"):
        build_toolchain(NAME, layout.root)


def test_a_missing_tool_is_refused_before_any_process_runs(layout: Layout) -> None:
    layout.translate.unlink()
    with pytest.raises(RunError, match="ttmlir-translate"):
        build_toolchain(NAME, layout.root)
    assert layout.log.calls == []


def test_a_tool_that_is_not_the_pinned_version_is_refused(layout: Layout) -> None:
    resolved = layout.root.resolve()
    layout.log.banners[str(resolved / layout.opt.relative_to(layout.root))] = "LLVM version 99.0.0 (PLACEHOLDER)\n"
    with pytest.raises(RunError, match="EXPECT_VERSION"):
        build_toolchain(NAME, layout.root)
