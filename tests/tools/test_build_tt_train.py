"""Tests for tools/build_tt_train.py and the tt-train-v0 data set it builds (2026-10-23; OQ-076 (1a)).

Bible: Training Module (Safeguards), Benchmark Suites (tt-pairs-v0, Tier A),
Agent Rules 1, 5, and 10.

The contract these tests fix:

- The run check: a run is run_lines consecutive non-blank lines, each
  stripped with inner whitespace collapsed, blank lines skipped; a record
  shares a run with a reference when any of its runs is one of the
  reference's runs. Every run counts, a run of braces too.
- build(files, spec, held, version) turns each included example into one
  record {prompt, completion} in example order: the prompt is the spec's
  template with the version, the task, and the completion's paths; the
  completion is the example's code files (.cpp, .hpp, .h under its
  directory, paths relative to it) as FILE blocks (lassi.core.files
  render_file_blocks). It refuses, naming the example: an included record
  that shares a run with a tracked Tier A reference (Agent Rule 5, rule 3);
  a completion file byte-identical to a file of a Tier A example directory,
  or sharing at least near_copy of its runs with those files (rule 4); an
  included or unlisted directory that is a Tier A example directory, or a
  held item not excluded (rule 1); and a code file under root that no
  listed directory covers. Two listed directories never cover one file:
  the deepest listed one does.
- The tracked spec, assets/train/tt-train-v0/dataset.yaml, lists every Tier
  A item's example directory under exclude, includes no matmul variant and
  no directory of a Tier A item, and the tracked manifest beside it matches
  it: its spec_sha256 is the spec's, its references are every tracked Tier
  A program with its current sha256 (a changed reference makes the
  manifest stale), its held items cover every eval and unassigned item of
  the suite, and its examples and exclusions are the spec's.
- THE TIER A TEST: when the generated records are present
  ($LASSI_SCRATCH/train-data/tt-train-v0.jsonl, the file the manifest pins
  by sha256), no record holds a run of 3 or more consecutive non-blank
  lines of a Tier A reference under assets/bench/tt-pairs-v0/. The test
  holds its own run check, so it does not trust the tool's. Without the
  records it skips, naming the build command; the manifest tests above run
  always.

The synthetic trees are SYNTHETIC; no value here is a measurement.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

from lassi.bench import load_suite
from lassi.bench.registry import HELD_SPLITS
from lassi.core.files import parse_file_blocks

REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "tools" / "build_tt_train.py"
SPEC = REPO / "assets" / "train" / "tt-train-v0" / "dataset.yaml"
MANIFEST = REPO / "assets" / "train" / "tt-train-v0" / "manifest.json"
SUITE = REPO / "assets" / "bench" / "tt-pairs-v0.yaml"
REFERENCES = REPO / "assets" / "bench" / "tt-pairs-v0"
RUN_LINES = 3


def tool() -> ModuleType:
    """Import tools/build_tt_train.py as the other tool tests import theirs."""
    spec = importlib.util.spec_from_file_location("build_tt_train_under_test", TOOL)
    assert spec is not None and spec.loader is not None, f"missing {TOOL}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# The test's own run check (independent of the tool's)


def own_runs(text: str, size: int = RUN_LINES) -> set[tuple[str, ...]]:
    """Return every run of `size` consecutive non-blank lines of `text`, each stripped and whitespace-collapsed."""
    lines = [" ".join(line.split()) for line in text.split("\n") if line.strip()]
    return {tuple(lines[index:index + size]) for index in range(len(lines) - size + 1)}


def tier_a_runs() -> dict[tuple[str, ...], str]:
    """Return every run of every tracked Tier A reference, with the reference that holds it."""
    found: dict[tuple[str, ...], str] = {}
    for path in sorted(REFERENCES.rglob("*")):
        if path.is_file():
            for run in own_runs(path.read_bytes().decode("utf-8")):
                found.setdefault(run, path.relative_to(REPO).as_posix())
    return found


def test_the_own_run_check_finds_a_planted_run_and_ignores_two_lines() -> None:
    reference = "int a = 1;\n\n  int   b = 2;\nint c = 3;\nint d = 4;\n"
    assert ("int a = 1;", "int b = 2;", "int c = 3;") in own_runs(reference)
    record = "x\nint a = 1;\nint b = 2;\n   int c = 3;\ny\n"
    assert own_runs(record) & own_runs(reference)
    assert not own_runs("int a = 1;\nint b = 2;\nz\n") & own_runs(reference)


# ---------------------------------------------------------------------------
# The tool's pieces, on SYNTHETIC trees

ROOT = "examples"
TEMPLATE = "Write it for $version.\nTask: $task\nFiles: $files.\n"


def spec_of(include: dict[str, Any], exclude: dict[str, str], **changes: Any) -> Any:
    """Return a tool Spec over the SYNTHETIC root with the given lists."""
    values: dict[str, Any] = {
        "dataset": "synthetic-set", "pin": "tt-metal", "root": ROOT, "suite": "assets/bench/synthetic.yaml",
        "run_lines": RUN_LINES, "near_copy": 0.5, "prompt": TEMPLATE,
        "include": {name: tool().Entry(task=task, described_from="SYNTHETIC")
                    for name, task in include.items()},
        "exclude": dict(exclude),
    }
    values.update(changes)
    return tool().Spec(**values)


def held_of(references: dict[str, str] | None = None, tier_dirs: dict[str, str] | None = None,
            items: tuple[str, ...] = ("held",)) -> Any:
    """Return a tool HeldOut: a SYNTHETIC suite with one held item whose example directory is examples/tier."""
    return tool().HeldOut(
        suite="synthetic", suite_path="assets/bench/synthetic.yaml", suite_sha256="0" * 64, items=items,
        tier_dirs={f"{ROOT}/tier": "held"} if tier_dirs is None else tier_dirs,
        references={"assets/bench/synthetic/held/tt/held.cpp": "REF one;\nREF two;\nREF three;\n"}
        if references is None else references,
    )


FILES = {
    f"{ROOT}/alpha/alpha.cpp": b"int main() {\n    alpha();\n    return 0;\n}\n",
    f"{ROOT}/alpha/kernels/k.cpp": b"void kernel_main() {\n    work();\n}\n",
    f"{ROOT}/alpha/CMakeLists.txt": b"SYNTHETIC\n",
    f"{ROOT}/alpha/README.md": b"# SYNTHETIC\n",
    f"{ROOT}/tier/tier.cpp": b"TIER one;\nTIER two;\nTIER three;\nTIER four;\n",
    f"{ROOT}/nested/beta/beta.cpp": b"int beta() {\n    beta_work();\n    return 1;\n}\n",
}


def test_build_writes_one_record_per_included_example() -> None:
    spec = spec_of({"alpha": "Do alpha.", "nested/beta": "Do beta."}, {"tier": "Tier A item held"})
    built = tool().build(FILES, spec, held_of(), "5280a9cf")
    assert [example["example"] for example in built.examples] == ["alpha", "nested/beta"]
    first = built.records[0]
    assert set(first) == {"prompt", "completion"}
    assert first["prompt"] == "Write it for 5280a9cf.\nTask: Do alpha.\nFiles: alpha.cpp, kernels/k.cpp.\n"
    parsed = parse_file_blocks(first["completion"])
    assert parsed.files == {"alpha.cpp": FILES[f"{ROOT}/alpha/alpha.cpp"].decode(),
                            "kernels/k.cpp": FILES[f"{ROOT}/alpha/kernels/k.cpp"].decode()}
    assert [entry["path"] for entry in built.examples[0]["files"]] == [f"{ROOT}/alpha/alpha.cpp",
                                                                       f"{ROOT}/alpha/kernels/k.cpp"]
    assert built.excluded[0]["example"] == "tier" and built.excluded[0]["reason"] == "Tier A item held"


def test_a_record_with_a_tracked_reference_run_is_refused() -> None:
    files = {**FILES, f"{ROOT}/alpha/alpha.cpp": b"x;\n  REF one;\nREF two;\n\nREF   three;\ny;\n"}
    spec = spec_of({"alpha": "Do alpha.", "nested/beta": "Do beta."}, {"tier": "Tier A item held"})
    with pytest.raises(ValueError) as refused:
        tool().build(files, spec, held_of(), "5280a9cf")
    message = str(refused.value)
    assert "alpha" in message and "assets/bench/synthetic/held/tt/held.cpp" in message and "REF one;" in message


def test_a_task_that_quotes_a_reference_run_is_refused_too() -> None:
    spec = spec_of({"alpha": "Do this:\nREF one;\nREF two;\nREF three;", "nested/beta": "Do beta."}, {"tier": "Tier A"})
    with pytest.raises(ValueError, match="alpha"):
        tool().build(FILES, spec, held_of(), "5280a9cf")


def test_a_copy_of_a_tier_a_file_is_refused() -> None:
    files = {**FILES, f"{ROOT}/alpha/kernels/k.cpp": FILES[f"{ROOT}/tier/tier.cpp"]}
    spec = spec_of({"alpha": "Do alpha.", "nested/beta": "Do beta."}, {"tier": "Tier A item held"})
    with pytest.raises(ValueError) as refused:
        tool().build(files, spec, held_of(), "5280a9cf")
    assert "alpha" in str(refused.value) and f"{ROOT}/tier/tier.cpp" in str(refused.value)


def test_a_near_copy_of_a_tier_a_file_is_refused() -> None:
    near = b"TIER one;\nTIER two;\nTIER three;\nTIER four;\nmine;\n"
    files = {**FILES, f"{ROOT}/alpha/kernels/k.cpp": near}
    spec = spec_of({"alpha": "Do alpha.", "nested/beta": "Do beta."}, {"tier": "Tier A item held"})
    with pytest.raises(ValueError, match="near copy"):
        tool().build(files, spec, held_of(), "5280a9cf")


def test_including_a_tier_a_directory_is_refused() -> None:
    spec = spec_of({"alpha": "Do alpha.", "nested/beta": "Do beta.", "tier": "Do tier."}, {})
    with pytest.raises(ValueError, match="Agent Rule 5"):
        tool().build(FILES, spec, held_of(), "5280a9cf")


def test_a_held_item_whose_directory_is_not_excluded_is_refused() -> None:
    spec = spec_of({"alpha": "Do alpha.", "nested/beta": "Do beta."}, {"other": "x"})
    files = {**FILES, f"{ROOT}/other/o.cpp": b"o;\n"}
    files.pop(f"{ROOT}/tier/tier.cpp")
    with pytest.raises(ValueError, match="held"):
        tool().build(files, spec, held_of(), "5280a9cf")


def test_an_unlisted_code_file_is_refused() -> None:
    spec = spec_of({"alpha": "Do alpha."}, {"tier": "Tier A item held"})
    with pytest.raises(ValueError, match="nested/beta/beta.cpp"):
        tool().build(FILES, spec, held_of(), "5280a9cf")


def test_the_deepest_listed_directory_covers_a_file() -> None:
    spec = spec_of({"alpha": "Do alpha.", "nested/beta": "Do beta."}, {"tier": "Tier A item held",
                                                                       "nested": "SYNTHETIC parent"})
    built = tool().build(FILES, spec, held_of(), "5280a9cf")
    assert "nested/beta" in [example["example"] for example in built.examples]
    nested = next(entry for entry in built.excluded if entry["example"] == "nested")
    assert nested["files"] == []


def test_the_records_text_is_ascii_json_lines_in_order() -> None:
    spec = spec_of({"alpha": "Do alpha.", "nested/beta": "Do beta."}, {"tier": "Tier A item held"})
    text = "int main() { /* " + chr(0xA9) + " */ }\n"
    built = tool().build({**FILES, f"{ROOT}/alpha/alpha.cpp": text.encode("utf-8")}, spec, held_of(), "5280a9cf")
    data = tool().records_bytes(built.records)
    assert data.isascii() and data.endswith(b"\n") and b"\r" not in data
    lines = data.decode("ascii").split("\n")[:-1]
    assert [json.loads(line) for line in lines] == list(built.records)


# ---------------------------------------------------------------------------
# The tracked spec and manifest


def spec_text() -> dict[str, Any]:
    """Return the tracked spec as YAML data."""
    return yaml.safe_load(SPEC.read_bytes().decode("ascii"))


def manifest() -> dict[str, Any]:
    """Return the tracked manifest."""
    assert MANIFEST.is_file(), f"{MANIFEST} is missing; run uv run tools/build_tt_train.py build --tree <tree>"
    return json.loads(MANIFEST.read_bytes().decode("ascii"))


def tier_a_dirs() -> dict[str, str]:
    """Return {example directory under the examples root: item} of every Tier A item."""
    found: dict[str, str] = {}
    for name, item in load_suite(SUITE).items.items():
        for language in item.languages.values():
            for entry in language.support.values():
                if not entry.tracked:
                    head = entry.path.partition("/kernels/")[0]
                    found[head.removeprefix("tt_metal/programming_examples/")] = name
    return found


def test_the_spec_excludes_every_tier_a_directory_and_every_matmul_variant() -> None:
    data = spec_text()
    for directory, item in tier_a_dirs().items():
        assert directory in data["exclude"], f"Tier A item {item}'s directory {directory} is not excluded"
        assert directory not in data["include"]
    assert not [name for name in data["include"] if "matmul" in name], "a matmul variant duplicates Tier A"
    assert not [name for name in data["include"] if "eltwise" in name or "vecadd" in name or "loopback" in name]


def test_the_loopback_duplicates_are_excluded_as_loopback_s_task() -> None:
    # Rule 2: a DRAM-to-DRAM move of a tile through L1 is loopback's task, whatever its route (2026-10-24 audit).
    exclude = spec_text()["exclude"]
    for name in ("NoC_tile_transfer", "profiler/test_noc_event_profiler"):
        assert name in exclude and "loopback" in exclude[name], name


KEPT_BESIDE = {
    # adds, as eltwise_binary does, but one uint32 on a data-movement core: no tile, compute kernel, or circular buffer
    "add_2_integers_in_riscv": "eltwise_binary",
    # reads into L1 and writes back, as loopback's kernel does, but 64 bytes to one remote core's address on the
    # experimental NoC API, with no DRAM buffer and no copy between two buffers, run as a profiler trace
    "profiler/test_device_api_debugger": "loopback",
    # writes a buffer and reads it back, as loopback's host does, but with no kernel: nothing on the device moves data
    "distributed/2_distributed_buffer_rw": "loopback",
}


@pytest.mark.parametrize(("example", "item"), list(KEPT_BESIDE.items()))
def test_an_example_kept_beside_a_tier_a_task_records_its_inclusion_judgment(example: str, item: str) -> None:
    # The spec says why each stays (the audits of 2026-10-24), and the manifest carries the judgment.
    entry = spec_text()["include"][example]
    assert "judgment" in entry and item in entry["judgment"], example
    found = next(record for record in manifest()["examples"] if record["example"] == example)
    assert found["judgment"] == " ".join(entry["judgment"].split())


def test_a_judgment_is_optional_and_reaches_the_examples_record() -> None:
    spec = spec_of({"alpha": "Do alpha.", "nested/beta": "Do beta."}, {"tier": "Tier A item held"})
    spec.include["alpha"] = tool().Entry(task="Do alpha.", described_from="SYNTHETIC", judgment="SYNTHETIC kept")
    built = tool().build(FILES, spec, held_of(), "5280a9cf")
    assert [example["judgment"] for example in built.examples] == ["SYNTHETIC kept", None]


def test_the_manifest_matches_the_spec_and_the_current_references() -> None:
    found = manifest()
    data = spec_text()
    assert found["corpus"] == data["dataset"] == "tt-train-v0"
    assert found["spec_sha256"] == hashlib.sha256(SPEC.read_bytes()).hexdigest(), "rebuild: the spec changed"
    references = {entry["path"]: entry["sha256"] for entry in found["held_out"]["references"]}
    current = {path.relative_to(REPO).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted(REFERENCES.rglob("*.cpp"))}
    assert references == current, "rebuild: a Tier A reference changed since the records were checked"
    suite = load_suite(SUITE)
    held = {name for name, item in suite.items.items() if item.split in HELD_SPLITS}
    assert held <= set(found["held_out"]["items"])
    assert found["held_out"]["suite"] == suite.name and found["held_out"]["run_lines"] == RUN_LINES
    assert [example["example"] for example in found["examples"]] == list(data["include"])
    assert [entry["example"] for entry in found["excluded"]] == list(data["exclude"])
    assert found["records"]["count"] == len(data["include"]) and found["records"]["file"] == "tt-train-v0.jsonl"
    assert found["source"]["commit"] == suite.commit


def test_the_manifest_records_each_source_file_and_no_tier_a_file_in_a_record() -> None:
    found = manifest()
    tier = {f"tt_metal/programming_examples/{directory}/" for directory in tier_a_dirs()}
    for example in found["examples"]:
        assert example["files"], example["example"]
        for entry in example["files"]:
            assert len(entry["sha256"]) == 64 and not any(entry["path"].startswith(prefix) for prefix in tier)


def generated_records() -> Path:
    """Return the generated records file, or skip naming the build command when it is absent."""
    scratch = os.environ.get("LASSI_SCRATCH", "")
    path = Path(scratch) / "train-data" / "tt-train-v0.jsonl" if scratch else None
    if path is None or not path.is_file():
        pytest.skip("no generated tt-train-v0 records under $LASSI_SCRATCH/train-data; build them with "
                    "uv run tools/build_tt_train.py build --tree <tree>")
    return path


def test_no_generated_record_holds_a_run_of_a_tier_a_reference() -> None:
    path = generated_records()
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == manifest()["records"]["sha256"], "the records are not the manifest's"
    runs = tier_a_runs()
    assert runs, "no Tier A reference was read"
    for number, line in enumerate(raw.decode("ascii").split("\n")[:-1], start=1):
        record = json.loads(line)
        for field in ("prompt", "completion"):
            shared = sorted(own_runs(record[field]) & set(runs))
            assert not shared, f"record {number} {field} holds {shared[0]} of {runs[shared[0]]}"
