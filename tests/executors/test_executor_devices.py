"""Tests for the device each executor names (task P4.5).

Bible: Execution Backends (none and native rows), Result Record
(provenance: device), Component Interfaces (Executor), Agent Rules 1 and 2.
PHASE-NOTES P2, "Device in native runs": provenance.json recorded device as
null for the native executor although the programs ran on the host CPU.

The contract these tests fix:

- Every registered Executor class has a method device() -> str: a
  non-empty, one-line, plain ASCII name of the device its programs run on.
  It starts no process and never touches the sandbox.
- NoneExecutor().device() is "none (compile only)", the value runs record
  today.
- NativeExecutor().device() names the host CPU: "host CPU (native): <model>"
  where <model> is the value of the first "model name" line of the file
  lassi.executors.native.CPUINFO (a Path; /proc/cpuinfo on Linux), with
  surrounding whitespace removed and every inner run of whitespace written
  as one space. When the file cannot be read, holds no "model name" line,
  or that value is empty, the device is "host CPU (native)" alone. The
  result is plain ASCII whatever the file holds.

The CPU model lines here are SYNTHETIC, written in the /proc/cpuinfo layout;
no host's CPU is read. No value in this module is a measurement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import lassi.executors  # noqa: F401  (importing the package registers every executor)
from lassi.core.registry import DEFAULT_REGISTRY
from lassi.executors import NativeExecutor, NoneExecutor
from lassi.executors import native as native_module

HOST_CPU = "host CPU (native)"
COMPILE_ONLY = "none (compile only)"
SYNTHETIC_MODEL = "SYNTHETIC CPU model 9000"


def cpuinfo(tmp_path: Path, text: str, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write `text` as a stand-in cpuinfo file and point lassi.executors.native.CPUINFO at it."""
    if not hasattr(native_module, "CPUINFO"):
        pytest.fail("lassi.executors.native has no CPUINFO; task P4.5 names the file the host CPU model is read from")
    path = tmp_path / "cpuinfo"
    path.write_bytes(text.encode("utf-8"))
    monkeypatch.setattr(native_module, "CPUINFO", path)
    return path


def device_of(executor: Any) -> str:
    """Return executor.device(); fail the test clearly while executors do not name their device."""
    method = getattr(executor, "device", None)
    if not callable(method):
        pytest.fail(f"{type(executor).__name__} has no device() method; task P4.5 makes each executor name its device")
    return method()


def block(model_line: str) -> str:
    """Return a SYNTHETIC /proc/cpuinfo block for one processor with `model_line` among its fields."""
    return f"processor\t: 0\nvendor_id\t: SyntheticVendor\ncpu family\t: 25\n{model_line}\nflags\t\t: fpu sse2\n\n"


def test_every_registered_executor_names_its_device() -> None:
    for name in DEFAULT_REGISTRY.names("Executor"):
        factory = DEFAULT_REGISTRY.get("Executor", name).factory
        assert callable(getattr(factory, "device", None)), f"Executor {name!r} has no device() method"


def test_the_compile_only_executor_names_no_device_it_runs_on() -> None:
    assert device_of(NoneExecutor()) == COMPILE_ONLY


def test_the_native_executor_names_the_host_cpu_and_its_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cpuinfo(tmp_path, block(f"model name\t: {SYNTHETIC_MODEL}"), monkeypatch)
    assert device_of(NativeExecutor()) == f"{HOST_CPU}: {SYNTHETIC_MODEL}"


def test_the_first_model_name_wins_and_whitespace_is_collapsed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    text = block("model name\t:   SYNTHETIC   CPU\tmodel  9000  ") + block("model name\t: SYNTHETIC other model")
    cpuinfo(tmp_path, text, monkeypatch)
    assert device_of(NativeExecutor()) == f"{HOST_CPU}: {SYNTHETIC_MODEL}"


@pytest.mark.parametrize(
    "text",
    [block("cpu model\t: SYNTHETIC field of another name"), block("model name\t:   "), ""],
    ids=["no-model-name-line", "empty-model-name", "empty-file"],
)
def test_without_a_model_name_the_device_is_the_host_cpu_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str
) -> None:
    cpuinfo(tmp_path, text, monkeypatch)
    assert device_of(NativeExecutor()) == HOST_CPU


def test_an_unreadable_cpuinfo_gives_the_host_cpu_alone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = cpuinfo(tmp_path, "", monkeypatch)
    path.unlink()
    assert device_of(NativeExecutor()) == HOST_CPU


def test_a_model_name_that_is_not_ascii_still_gives_a_plain_ascii_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cpuinfo(tmp_path, block("model name\t: SYNTHETIC CPU \N{GREEK SMALL LETTER MU}-model"), monkeypatch)
    device = device_of(NativeExecutor())
    assert device.isascii() and device.startswith(HOST_CPU)
    assert "\n" not in device


class ExplodingSandbox:
    """A stand-in sandbox: any use of it fails the test, since naming a device runs nothing."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"device() used the sandbox ({name})")


def test_naming_the_device_runs_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cpuinfo(tmp_path, block(f"model name\t: {SYNTHETIC_MODEL}"), monkeypatch)
    executor = NativeExecutor(sandbox=ExplodingSandbox())  # type: ignore[arg-type]
    assert device_of(executor) == f"{HOST_CPU}: {SYNTHETIC_MODEL}"


@pytest.mark.parametrize("executor", [NoneExecutor, NativeExecutor], ids=["none", "native"])
def test_a_device_is_one_line_of_plain_ascii(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, executor: type
) -> None:
    path = tmp_path / "cpuinfo"
    path.write_bytes(block(f"model name\t: {SYNTHETIC_MODEL}").encode("ascii"))
    monkeypatch.setattr(native_module, "CPUINFO", path, raising=False)
    device = device_of(executor())
    assert device and device.isascii() and device == device.strip() and "\n" not in device


def test_a_control_character_in_the_model_becomes_an_escape_and_whitespace_collapses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A SYNTHETIC model value with a BEL character (not whitespace, not printable) between blanks and a tab.
    cpuinfo(tmp_path, block("model name\t:  SYNTHETIC \x07 CPU\t9000 "), monkeypatch)
    device = device_of(NativeExecutor())
    assert device == f"{HOST_CPU}: SYNTHETIC \\x07 CPU 9000"
    assert device.isascii() and device.isprintable() and device == device.strip()
