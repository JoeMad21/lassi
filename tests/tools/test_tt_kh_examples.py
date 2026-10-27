"""Tests for tt-kh-examples-v0, the worked examples a C++ -> TT prompt that asks for kernels carries (task P17.39).

Bible: Project Recipes (Notes, the Examples bullet), Benchmark Suites
(tt-pairs-v0, a development set, OQ-083 (b)), Agent Rules 5 and 10.

tools/build_tt_examples.py builds it from assets/examples/tt-kh-examples-v0/spec.yaml by the rules that build
tt-examples-v0: every listed directory is in tt-train-v0's include list (so never a Tier A item or a duplicate of
one), every code file of each directory is listed, the host program first and then its kernels, sorted, and no
file holds a run of a tracked Tier A reference or copies a Tier A example file. What these tests fix:

- The spec names the pack, the tt-metal pin, and tt-train-v0's data set spec, and lists, in order,
  hello_world_compute_kernel, hello_world_datatypes_kernel, add_2_integers_in_riscv, and shard_data_rm, all in the
  include list: between them a compute kernel (its entry point and include at the pin) and data movement kernels
  that use a circular buffer and NoC reads and writes.
- The tracked manifest matches the spec and the pin, lists every file with its sha256 and size in spec order, and
  its sha256 is that of its entries; only the spec and the manifest are tracked.
- When the fragments are built here they match the manifest, hold no run of 3 lines of a tracked Tier A reference,
  and check mode passes on them.

No value here is a measurement.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

from lassi.prompts import examples as examples_module
from lassi.toolchains import pins

REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "tools" / "build_tt_examples.py"
PACK = "tt-kh-examples-v0"
PACK_DIR = REPO / "assets" / "examples" / PACK
SPEC = PACK_DIR / "spec.yaml"
MANIFEST = PACK_DIR / "MANIFEST.yaml"
DATASET = REPO / "assets" / "train" / "tt-train-v0" / "dataset.yaml"
REFERENCES = REPO / "assets" / "bench" / "tt-pairs-v0"
EXPECTED = ["hello_world_compute_kernel", "hello_world_datatypes_kernel", "add_2_integers_in_riscv", "shard_data_rm"]


def tool() -> ModuleType:
    """Import tools/build_tt_examples.py as the other tool tests import theirs."""
    spec = importlib.util.spec_from_file_location("build_tt_examples_for_kh", TOOL)
    assert spec is not None and spec.loader is not None, f"missing {TOOL}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(data: bytes) -> str:
    """Return the sha256 hex digest of `data`."""
    return hashlib.sha256(data).hexdigest()


def spec_data() -> dict[str, Any]:
    """Return the tracked spec."""
    return yaml.safe_load(SPEC.read_bytes().decode("ascii"))


def manifest_data() -> dict[str, Any]:
    """Return the tracked manifest."""
    assert MANIFEST.is_file(), f"{MANIFEST} is missing; run uv run tools/build_tt_examples.py build --spec {SPEC}"
    return yaml.safe_load(MANIFEST.read_bytes().decode("ascii"))


def test_the_spec_lists_four_included_examples_with_compute_and_data_movement_kernels() -> None:
    data = spec_data()
    dataset = yaml.safe_load(DATASET.read_bytes().decode("ascii"))
    assert data["pack"] == PACK and data["pin"] == dataset["pin"] == "tt-metal"
    assert data["dataset"] == DATASET.relative_to(REPO).as_posix()
    assert [entry["directory"] for entry in data["examples"]] == EXPECTED
    kernels = []
    for entry in data["examples"]:
        assert entry["directory"] in dataset["include"] and entry["directory"] not in dataset["exclude"]
        host, *listed = entry["files"]
        assert host.endswith(".cpp") and not host.startswith("kernels/"), entry
        assert listed and listed == sorted(listed) and all(path.startswith("kernels/") for path in listed)
        kernels += listed
    assert any(path.startswith("kernels/compute/") for path in kernels), "a compute kernel at the pin"
    assert any(path.startswith("kernels/dataflow/") for path in kernels), "a data movement kernel with a CB"
    assert [example.directory for example in tool().load_spec(SPEC).examples] == EXPECTED


def test_the_tracked_manifest_matches_the_spec_and_the_pin() -> None:
    found = manifest_data()
    pin = pins.read_pin("tt-metal")
    assert found["generator"] == "tools/build_tt_examples.py" and found["pack"] == PACK
    assert (found["repo"], found["commit"]) == (pin["URL"], pin["COMMIT"])
    assert found["spec"] == SPEC.relative_to(REPO).as_posix()
    assert found["spec_sha256"] == sha256(SPEC.read_bytes()), "rebuild: the spec changed"
    listed = [[f"{entry['directory']}/{path}" for path in entry["files"]] for entry in spec_data()["examples"]]
    assert [[row["path"] for row in example["files"]] for example in found["examples"]] == listed
    rows = [row for example in found["examples"] for row in example["files"]]
    assert all(len(row["sha256"]) == 64 and row["bytes"] > 0 for row in rows)
    lines = "".join(f"{row['sha256']}  {row['path']}\n" for row in rows)
    assert found["sha256"] == sha256(lines.encode("ascii"))


def test_only_the_spec_and_the_manifest_are_tracked() -> None:
    lines = (REPO / ".gitignore").read_bytes().decode("ascii").splitlines()
    for line in (f"/assets/examples/{PACK}/*", f"!/assets/examples/{PACK}/MANIFEST.yaml",
                 f"!/assets/examples/{PACK}/spec.yaml"):
        assert line in lines, f".gitignore lacks {line}"


def own_runs(text: str, size: int = 3) -> set[tuple[str, ...]]:
    """Return every run of `size` consecutive non-blank lines of `text`, each stripped and whitespace-collapsed."""
    lines = [" ".join(line.split()) for line in text.split("\n") if line.strip()]
    return {tuple(lines[index:index + size]) for index in range(len(lines) - size + 1)}


def built_fragments() -> list[Path]:
    """Return the generated fragments the manifest lists, or skip naming the build command when any is absent."""
    paths = [PACK_DIR / row["path"] for example in manifest_data()["examples"] for row in example["files"]]
    if not all(path.is_file() for path in paths):
        pytest.skip(f"the {PACK} fragments are not built; build them with uv run tools/build_tt_examples.py fetch "
                    f"--dest <dir> --spec {SPEC.relative_to(REPO).as_posix()}, then build --tree <dir> --spec ...")
    return paths


def test_the_built_fragments_match_the_manifest_and_hold_no_tier_a_run() -> None:
    paths = built_fragments()
    assert examples_module.check_pack(PACK_DIR) == []
    references: dict[tuple[str, ...], str] = {}
    for path in sorted(REFERENCES.rglob("*")):
        if path.is_file():
            for run in own_runs(path.read_bytes().decode("utf-8")):
                references.setdefault(run, path.relative_to(REPO).as_posix())
    for path in paths:
        shared = sorted(own_runs(path.read_bytes().decode("utf-8")) & set(references))
        assert not shared, f"{path.name} holds {shared[0]} of {references[shared[0]]}"
    compute = [path for path in paths if "/kernels/compute/" in path.as_posix()]
    assert compute and all("void kernel_main()" in path.read_text(encoding="utf-8") for path in compute)


def test_check_mode_passes_on_the_built_pack() -> None:
    built_fragments()
    assert tool().main(["check", "--spec", str(SPEC)]) == 0
