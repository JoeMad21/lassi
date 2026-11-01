"""Tests for the record field of each pin a run declares (task P17.48; lassi.core.runner pin_field).

Bible: Toolchain Pins, the Trial record (toolchain_pins), Agent Rule 10.

The contract these tests fix:

- A pin is recorded under the toolchain_pins field of its name, "-" written
  "_" (tt-metal: tt_metal), as before.
- A build variant in a pin file of its own, <NAME>-<BUILD>.pin with those
  NAME and BUILD keys, is recorded under NAME's field as <VERSION>-<BUILD>:
  toolchains/tt-mlir-runtime.pin (NAME tt-mlir, BUILD runtime) as tt_mlir,
  0.9.0.dev20260221-runtime. Any other pin without a field is a RunError
  naming it (the driver pin among them).
- The same pin name declared twice records one value; two pin names that
  would put two versions in one field (the compiler pin tt-mlir beside the
  runtime build) are a RunError naming both, since a trial records one
  version of each field.

The pins are the tracked pin files, read as they are; nothing is installed or
run.
"""

from __future__ import annotations

import pytest

from lassi.core.record import ToolchainPins
from lassi.core.runner import RunError, _pin_versions, pin_field
from lassi.toolchains.pins import read_pin


def test_a_pin_is_recorded_under_the_field_of_its_name() -> None:
    assert pin_field("tt-metal", read_pin("tt-metal")) == "tt_metal"
    assert pin_field("tt-mlir", read_pin("tt-mlir")) == "tt_mlir"
    assert pin_field("ttsim", read_pin("ttsim")) == "ttsim"


def test_the_runtime_build_is_recorded_under_tt_mlir() -> None:
    runtime = read_pin("tt-mlir-runtime")
    assert (runtime["NAME"], runtime["BUILD"]) == ("tt-mlir", "runtime")
    assert pin_field("tt-mlir-runtime", runtime) == "tt_mlir"
    pins = _pin_versions([{"tt-mlir-runtime": runtime, "ttsim": read_pin("ttsim")}])
    assert pins == ToolchainPins(tt_mlir="0.9.0.dev20260221-runtime", ttsim=read_pin("ttsim")["VERSION"])


@pytest.mark.parametrize(
    ("name", "pin"),
    [
        ("tt-mlir-driver", read_pin("tt-mlir-driver")),
        ("tt-mlir-runtime", {"NAME": "tt-mlir", "VERSION": "1"}),
        ("tt-mlir-other", {"NAME": "tt-mlir", "BUILD": "runtime", "VERSION": "1"}),
        ("foo-runtime", {"NAME": "foo", "BUILD": "runtime", "VERSION": "1"}),
    ],
    ids=["driver", "no-build", "other-stem", "no-source-field"],
)
def test_a_pin_without_a_field_is_refused_naming_it(name: str, pin: dict[str, str]) -> None:
    with pytest.raises(RunError, match=name):
        pin_field(name, pin)


def test_one_pin_declared_by_a_toolchain_and_an_executor_records_one_value() -> None:
    runtime = read_pin("tt-mlir-runtime")
    pins = _pin_versions([{"tt-mlir-runtime": runtime}, {"tt-mlir-runtime": dict(runtime)}])
    assert pins.tt_mlir == "0.9.0.dev20260221-runtime"


def test_the_compiler_pin_beside_the_runtime_build_is_refused_naming_both() -> None:
    with pytest.raises(RunError, match="'tt-mlir'.*'tt-mlir-runtime'.*tt_mlir"):
        _pin_versions([{"tt-mlir": read_pin("tt-mlir")}, {"tt-mlir-runtime": read_pin("tt-mlir-runtime")}])
