"""Tests for the ttsim presets of the TTIR and D2M toolchains (task P17.48; plans/spikes/ttmlir-runtime.md).

Bible: IR Strategy (TTIR and D2M), Toolchain Pins (the tt-mlir runtime
build), Execution Backends (the ttsim row), Agent Rules 2, 3, 6, and 10.

The design these tests fix. The Toolchains "ttir-ttmetal-ttsim" and
"d2m-ttmetal-ttsim" (lassi.toolchains.ttir_compile.TtirTtsimCompile and
D2mTtsimCompile) build as ttir-ttmetal-compile and d2m-ttmetal-compile do,
with two changes: their pin is tt-mlir-runtime (PIN, PIN_BIN TTMLIR_OPT, and
TOOL_PINS TTMLIR_TRANSLATE), the pin the ttmlir-ttsim executor declares; and
the lowering is

    ttmlir-opt --ttir-to-ttmetal-pipeline=system-desc-path=<descriptor>
        enable-multicast-inference=false <m>.mlir -o _ttmetal/<m>.ttmetal.mlir

with <descriptor> = <toolchains root>/<driver PREFIX_NAME>/<SYSTEM_DESC>, the
root read off the pinned ttmlir-opt's path. check_tool_pin refuses an
install whose record does not start with the runtime pin's "<NAME> <COMMIT>
<URL> <VERSION> <BUILD>", and a descriptor that is missing, has another
sha256, or has none recorded (PLACEHOLDER). The compile-only presets keep
their pin and their lowering.

No tool runs: a fake command runner answers each command and writes the
files a real run would; every module and descriptor here is SYNTHETIC.
"""

from __future__ import annotations

import hashlib
import re
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

import lassi.toolchains  # noqa: F401  (importing the package registers every toolchain preset)
from lassi.core.registry import DEFAULT_REGISTRY
from lassi.toolchains import CommandResult, ttir_compile
from lassi.toolchains import pins as pins_module

REPO = Path(__file__).resolve().parents[2]
NAMES = {"ttir": "ttir-ttmetal-ttsim", "d2m": "d2m-ttmetal-ttsim"}
COMPILE_ONLY = {"ttir": "ttir-ttmetal-compile", "d2m": "d2m-ttmetal-compile"}
MODULE = "probe.mlir"
DESC_BYTES = b"SYNTHETIC: stands in for the recorded ttsim system descriptor in tests\n"
TTIR_TEXT = (
    "// SYNTHETIC TTIR module written for this test\n"
    "func.func @probe(%x: tensor<64x128xbf16>, %y: tensor<64x128xbf16>) -> tensor<64x128xbf16> {\n"
    '  %r = "ttir.add"(%x, %y) : (tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>\n'
    "  return %r : tensor<64x128xbf16>\n"
    "}\n"
)
D2M_TEXT = "// SYNTHETIC D2M module written for this test\nmodule {\n  // d2m.generic ...\n}\n"
FLATBUFFER = b"\x3c\xcc\x00\x00\x20\x00\x00\x00TTM0" + bytes(16)


@dataclass
class FakeTools:
    """A CommandRunner standing in for ttmlir-opt and ttmlir-translate: it records each call and writes its -o file."""

    printed: str
    calls: list[list[str]] = field(default_factory=list)

    def __call__(self, argv: Sequence[str], cwd: Path, timeout_s: float) -> CommandResult:
        """Record the call and write what the step would write."""
        self.calls.append(list(argv))
        out = Path(cwd) / argv[argv.index("-o") + 1]
        out.parent.mkdir(parents=True, exist_ok=True)
        if any(arg.startswith("--ttmetal-to-flatbuffer") for arg in argv):
            out.write_bytes(FLATBUFFER)
        elif any(arg.startswith("--ttir-to-ttmetal-pipeline") for arg in argv):
            out.write_bytes(b"module {\n  ttmetal.enqueue_program\n}\n")
        else:
            out.write_bytes(self.printed.encode())
        return CommandResult(0, "", "")


@dataclass(frozen=True)
class Install:
    """The SYNTHETIC toolchains root: the runtime build's tools and record, and the driver prefix's descriptor."""

    root: Path
    opt: Path
    translate: Path
    record: Path
    desc: Path


def rewrite(path: Path, values: dict[str, str]) -> None:
    """Replace the KEY=value line of each key in `values` in a pin file."""
    text = path.read_bytes().decode("ascii")
    for key, value in values.items():
        text, count = re.subn(rf"(?m)^{key}=.*$", f"{key}={value}", text)
        assert count == 1, key
    path.write_bytes(text.encode("ascii"))


@pytest.fixture
def install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Install:
    """Point the pin reader at a copy of toolchains/ whose descriptor sha256 is DESC_BYTES', and lay out the root."""
    pins_dir = tmp_path / "pins"
    shutil.copytree(REPO / "toolchains", pins_dir, ignore=shutil.ignore_patterns("tt-mlir-driver"))
    rewrite(pins_dir / "tt-mlir-driver.pin", {"SYSTEM_DESC_SHA256": hashlib.sha256(DESC_BYTES).hexdigest()})
    monkeypatch.setattr(pins_module, "PINS_DIR", pins_dir)
    runtime, driver = pins_module.read_pin("tt-mlir-runtime"), pins_module.read_pin("tt-mlir-driver")
    root = tmp_path / "toolchains"
    prefix = root / runtime["PREFIX_NAME"]
    opt, translate = prefix / runtime["TTMLIR_OPT"], prefix / runtime["TTMLIR_TRANSLATE"]
    opt.parent.mkdir(parents=True)
    opt.write_bytes(b"SYNTHETIC tool, never run\n")
    translate.write_bytes(b"SYNTHETIC tool, never run\n")
    head = " ".join(runtime[key] for key in ("NAME", "COMMIT", "URL", "VERSION", "BUILD"))
    record = prefix / "lassi-install.txt"
    record.write_bytes(f"{head}\nversion-tag v0.0\n".encode("ascii"))
    desc = root / driver["PREFIX_NAME"] / driver["SYSTEM_DESC"]
    desc.parent.mkdir(parents=True)
    desc.write_bytes(DESC_BYTES)
    return Install(root, opt, translate, record, desc)


def toolchain(language: str, opt: Path, translate: Path, runner: FakeTools) -> Any:
    """Return the language's ttsim preset built as the runner builds it."""
    cls = DEFAULT_REGISTRY.get("Toolchain", NAMES[language]).factory
    return cls(executable=str(opt), runner=runner, tools={"TTMLIR_TRANSLATE": str(translate)})


# ---------------------------------------------------------------------------
# Registration and pins


@pytest.mark.parametrize("language", sorted(NAMES))
def test_each_preset_is_registered_on_the_runtime_pin(language: str) -> None:
    cls = DEFAULT_REGISTRY.get("Toolchain", NAMES[language]).factory
    base = DEFAULT_REGISTRY.get("Toolchain", COMPILE_ONLY[language]).factory
    assert cls.__module__ == "lassi.toolchains.ttir_compile" and cls.name == NAMES[language]
    assert issubclass(cls, base) and cls.LANGUAGE == base.LANGUAGE and cls.BACKEND is ttir_compile.TTMETAL
    assert cls.capabilities == base.capabilities
    pin = pins_module.read_pin("tt-mlir-runtime")
    assert cls.PIN == "tt-mlir-runtime" and cls.PIN_BIN.format_map(pin) == pin["TTMLIR_OPT"]
    assert dict(cls.TOOL_PINS) == {"tt-mlir-runtime": ("TTMLIR_TRANSLATE",)}
    assert base.PIN == "tt-mlir" and dict(base.TOOL_PINS) == {"tt-mlir": ("TTMLIR_TRANSLATE",)}, "unchanged"


def test_the_executor_declares_the_same_pin() -> None:
    from lassi.executors.ttmlir_ttsim import TtmlirTtsimExecutor

    cls = DEFAULT_REGISTRY.get("Toolchain", NAMES["ttir"]).factory
    assert TtmlirTtsimExecutor().pins[cls.PIN] == pins_module.read_pin(cls.PIN)


def test_the_root_is_read_off_the_pinned_tool() -> None:
    pin = pins_module.read_pin("tt-mlir-runtime")
    root = Path("/PLACEHOLDER/toolchains").resolve()
    opt = root / pin["PREFIX_NAME"] / pin["TTMLIR_OPT"]
    assert ttir_compile.toolchains_root_of(str(opt), pin) == root
    assert ttir_compile.toolchains_root_of("ttmlir-opt", pin) is None
    assert ttir_compile.toolchains_root_of(str(root / "elsewhere" / pin["TTMLIR_OPT"]), pin) is None


# ---------------------------------------------------------------------------
# The build


@pytest.mark.parametrize(("language", "text"), [("ttir", TTIR_TEXT), ("d2m", D2M_TEXT)])
def test_the_lowering_names_the_recorded_descriptor(install: Install, tmp_path: Path, language: str, text: str) -> None:
    runner = FakeTools(printed=TTIR_TEXT if language == "ttir" else "module {\n  d2m.generic\n}\n")
    workdir = tmp_path / "build"
    workdir.mkdir()
    built = toolchain(language, install.opt, install.translate, runner)
    assert built.system_desc == install.desc
    result = built.build({MODULE: text}, workdir)
    verify, lower, translate = runner.calls
    pipeline = f"--ttir-to-ttmetal-pipeline=system-desc-path={install.desc} enable-multicast-inference=false"
    assert lower == [str(install.opt), pipeline, MODULE, "-o", "_ttmetal/probe.ttmetal.mlir"]
    assert verify[0] == str(install.opt) and translate[:2] == [str(install.translate), "--ttmetal-to-flatbuffer"]
    assert result.artifact == workdir / "_ttmetal" / "probe.ttm"
    assert [item for item in result.diagnostics if item.severity == "error"] == []


def test_a_given_descriptor_wins_and_none_is_refused_before_any_command(tmp_path: Path) -> None:
    cls = DEFAULT_REGISTRY.get("Toolchain", NAMES["ttir"]).factory
    given = cls(executable="ttmlir-opt", system_desc="/PLACEHOLDER/ttsim.ttsys")
    expected = f"--ttir-to-ttmetal-pipeline=system-desc-path={Path('/PLACEHOLDER/ttsim.ttsys')} "
    assert given.lowering_pipeline().startswith(expected)
    runner = FakeTools(printed=TTIR_TEXT)
    bare = cls(executable="ttmlir-opt", runner=runner)
    assert bare.system_desc is None
    with pytest.raises(ValueError, match="no ttsim system descriptor"):
        bare.lowering_pipeline()
    workdir = tmp_path / "build"
    workdir.mkdir()
    with pytest.raises(ValueError, match="no ttsim system descriptor"):
        bare.build({MODULE: TTIR_TEXT}, workdir)
    assert runner.calls == []


def test_the_compile_only_presets_keep_the_mock_descriptor() -> None:
    for name in COMPILE_ONLY.values():
        built = DEFAULT_REGISTRY.get("Toolchain", name).factory()
        assert built.lowering_pipeline() == "--ttir-to-ttmetal-pipeline=enable-multicast-inference=false"


# ---------------------------------------------------------------------------
# check_tool_pin


def check(install: Install, pin_name: str = "tt-mlir-runtime") -> None:
    """Run the ttir preset's check_tool_pin on the runtime prefix."""
    cls = DEFAULT_REGISTRY.get("Toolchain", NAMES["ttir"]).factory
    cls.check_tool_pin(pin_name, install.record.parent, pins_module.read_pin("tt-mlir-runtime"))


def test_the_installed_runtime_and_descriptor_pass(install: Install) -> None:
    check(install)


def test_another_pin_name_is_refused(install: Install) -> None:
    with pytest.raises(ValueError, match="tool pin"):
        check(install, "tt-mlir")


def test_the_compiler_installs_record_is_refused(install: Install) -> None:
    compiler = pins_module.read_pin("tt-mlir")
    install.record.write_bytes(f"{compiler['NAME']} {compiler['COMMIT']} {compiler['URL']} {compiler['VERSION']}\n"
                               .encode("ascii"))
    with pytest.raises(ValueError, match="runtime build"):
        check(install)


def test_a_missing_record_is_refused(install: Install) -> None:
    install.record.unlink()
    with pytest.raises(ValueError, match="cannot be read"):
        check(install)


def test_a_descriptor_with_other_bytes_is_refused(install: Install) -> None:
    install.desc.write_bytes(b"SYNTHETIC: another descriptor\n")
    with pytest.raises(ValueError, match="sha256"):
        check(install)


def test_a_missing_descriptor_is_refused(install: Install) -> None:
    install.desc.unlink()
    with pytest.raises(ValueError, match="missing"):
        check(install)


def test_no_recorded_descriptor_is_refused(install: Install) -> None:
    rewrite(pins_module.PINS_DIR / "tt-mlir-driver.pin", {"SYSTEM_DESC_SHA256": "PLACEHOLDER"})
    with pytest.raises(ValueError, match="PLACEHOLDER"):
        check(install)


def test_a_descriptor_path_with_whitespace_is_refused(tmp_path: Path) -> None:
    spaced = tmp_path / "a b" / "ttsim.ttsys"
    spaced.parent.mkdir()
    spaced.write_bytes(DESC_BYTES)
    pin = {"SYSTEM_DESC_SHA256": hashlib.sha256(DESC_BYTES).hexdigest()}
    with pytest.raises(ValueError, match="whitespace"):
        ttir_compile.check_system_desc(spaced, pin)
