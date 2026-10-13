"""Tests for the training pins in pyproject.toml and uv.lock (task P17.9).

Bible: Toolchain Pins (framework pins: the extras cpu, cuda, and rocm, one
per torch build, which conflict; a separate train extra was refused in task
P17.4, since alone it resolves PyPI's torch); plans/p17-portable.md, task
P17.9 ("The train extra's pins go into pyproject.toml and uv.lock");
PHASE-NOTES P17, "P17.9 plan wording".

The contract these tests fix:

- Each of the extras cpu, cuda, and rocm holds exactly torch 2.14.1,
  transformers 5.18.0, trl 1.14.1, peft 0.21.2, and accelerate 1.15.0
  (rocm also keeps triton-rocm 3.8.0 on Linux); there is no other extra, so
  no train extra; the project's own dependencies stay the control plane.
- The torch indexes do not change: three explicit PyTorch indexes (whl/cpu,
  whl/cu130, whl/rocm7.2), scoped to torch and triton-rocm through
  tool.uv.sources, with the three extras in one conflict group; trl, peft,
  and accelerate come from PyPI.
- uv.lock holds one entry each of trl 1.14.1, peft 0.21.2, and accelerate
  1.15.0 from PyPI, records them in lassi's optional dependencies for each
  extra, and requires them at those pins for each extra.

These tests read the two files and run nothing. No value in this module is
a measurement.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
from train_fakes import REPO

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

PYPROJECT = REPO / "pyproject.toml"
UV_LOCK = REPO / "uv.lock"
FLAVORS = ("cpu", "cuda", "rocm")
PYPI = "https://pypi.org/simple"
TRAIN_PINS = {"trl": "1.14.1", "peft": "0.21.2", "accelerate": "1.15.0"}
FLAVOR_PINS = {"torch": "2.14.1", "transformers": "5.18.0", **TRAIN_PINS}
TRITON = "triton-rocm==3.8.0;sys_platform=='linux'"
CONTROL_PLANE = {"pyarrow", "pyyaml", "tiktoken"}
INDEXES = {
    "pytorch-cpu": "https://download.pytorch.org/whl/cpu",
    "pytorch-cu130": "https://download.pytorch.org/whl/cu130",
    "pytorch-rocm": "https://download.pytorch.org/whl/rocm7.2",
}
TORCH_SOURCES = [
    {"index": "pytorch-cpu", "extra": "cpu"},
    {"index": "pytorch-cu130", "extra": "cuda", "marker": "sys_platform == 'linux' or sys_platform == 'win32'"},
    {"index": "pytorch-rocm", "extra": "rocm", "marker": "sys_platform == 'linux'"},
]


def read_toml(path: Path) -> dict[str, Any]:
    """Return a TOML file of the repository, read as ASCII."""
    return tomllib.loads(path.read_bytes().decode("ascii"))


def lock_packages() -> dict[str, list[dict[str, Any]]]:
    """Return uv.lock's package entries by name."""
    found: dict[str, list[dict[str, Any]]] = {}
    for package in read_toml(UV_LOCK)["package"]:
        found.setdefault(package["name"], []).append(package)
    return found


def test_each_flavor_extra_holds_the_training_pins() -> None:
    extras = read_toml(PYPROJECT)["project"]["optional-dependencies"]
    assert sorted(extras) == sorted(FLAVORS), "the training pins join the flavor extras; there is no train extra"
    expected = {f"{name}=={version}" for name, version in FLAVOR_PINS.items()}
    for flavor in FLAVORS:
        found = {item.replace(" ", "") for item in extras[flavor]}
        wanted = expected | ({TRITON} if flavor == "rocm" else set())
        assert found == wanted, (flavor, sorted(found ^ wanted))


def test_the_control_plane_and_the_torch_indexes_do_not_change() -> None:
    project = read_toml(PYPROJECT)
    names = {item.split(">")[0].split("=")[0].strip() for item in project["project"]["dependencies"]}
    assert names == CONTROL_PLANE, names
    uv = project["tool"]["uv"]
    assert [sorted(item["extra"] for item in group) for group in uv["conflicts"]] == [sorted(FLAVORS)]
    assert sorted(uv["sources"]) == ["torch", "triton-rocm"], "trl, peft, and accelerate come from PyPI"
    assert uv["sources"]["torch"] == TORCH_SOURCES
    assert uv["sources"]["triton-rocm"] == [
        {"index": "pytorch-rocm", "extra": "rocm", "marker": "sys_platform == 'linux'"}
    ]
    indexes = {entry["name"]: entry for entry in uv["index"]}
    assert {name: entry["url"] for name, entry in indexes.items()} == INDEXES
    assert all(entry.get("explicit") is True for entry in indexes.values())


@pytest.mark.parametrize("name", list(TRAIN_PINS))
def test_the_lock_holds_one_pinned_entry_from_pypi(name: str) -> None:
    entries = lock_packages().get(name, [])
    assert [entry["version"] for entry in entries] == [TRAIN_PINS[name]], f"uv.lock locks {name}: {entries}"
    assert entries[0]["source"] == {"registry": PYPI}


def test_the_lock_records_the_pins_for_each_flavor() -> None:
    (own,) = lock_packages()["lassi"]
    assert {item["name"] for item in own["dependencies"]} == CONTROL_PLANE
    optional = own["optional-dependencies"]
    requires = own["metadata"]["requires-dist"]
    for flavor in FLAVORS:
        names = {item["name"] for item in optional[flavor]}
        assert set(TRAIN_PINS) <= names, (flavor, sorted(names))
        for name, version in TRAIN_PINS.items():
            matching = [
                item for item in requires
                if item["name"] == name and item.get("marker") == f"extra == '{flavor}'"
            ]
            assert [item["specifier"] for item in matching] == [f"=={version}"], (flavor, name, matching)
    assert own["metadata"]["provides-extras"] == list(FLAVORS)


def test_the_lock_still_resolves_each_torch_build() -> None:
    versions = {entry["version"] for entry in lock_packages()["torch"]}
    assert {"2.14.1+cpu", "2.14.1+cu130", "2.14.1+rocm7.2"} <= versions, sorted(versions)
