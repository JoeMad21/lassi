"""Tests for tools/build_tt_examples.py and the tt-examples-v0 pack it builds (OQ-083 (b), 2026-10-25).

Bible: Benchmark Suites (tt-pairs-v0, Tier A), Agent Rules 1, 5, and 10.

The contract these tests fix:

- The spec (assets/examples/tt-examples-v0/spec.yaml) names the pack, the
  pin, the data set spec whose include and exclude lists it is held to
  (assets/train/tt-train-v0/dataset.yaml), and, in order, each example
  directory with every code file under it: the host program first, then
  its kernels (under kernels/), sorted. It lists exactly shard_data_rm and
  add_2_integers_in_riscv.
- build(files, spec, dataset, held) returns the fragments, each pack path
  <directory>/<file> with exactly the tree's bytes, in spec order. It
  refuses, naming the directory or file: a directory not in the data set's
  include list (one in its exclude list among them, a Tier A directory
  first); a listed file that is not a code file the directory holds; a
  code file of the directory left unlisted; a host program not first or
  kernels not sorted; a held item's example directory with no code file in
  the tree (rule 4 would read nothing); and the data set's rule 3 and rule
  4 (build_tt_train.py) on every file: a run of run_lines lines of a
  tracked Tier A reference, a byte-identical copy of a Tier A example file,
  or a near copy of those files.
- fetch downloads, at the pin's commit, each file under a listed
  directory and under each held item's example directory, each checked
  against its git blob sha1, into a directory outside the repository, and
  leaves a record naming the commit and the pack; nothing else is fetched.
- The manifest (MANIFEST.yaml, tracked) names the generator, the repo, the
  commit, every file's pack path, sha256, and size, in spec order, and the
  pack's sha256, that of the lines "<sha256>  <path>\\n" of every file in
  order. lassi.prompts.examples check_pack names each fragment missing,
  of another size, or of another sha256, and a manifest whose pack sha256
  is not its entries'; load_pack returns the examples in order, each file
  at its path within its directory.
- The fragments are gitignored; only spec.yaml and MANIFEST.yaml are
  tracked. When the fragments are present they match the manifest, and no
  fragment holds a run of 3 consecutive non-blank lines of a tracked Tier A
  reference (the test's own run check, as test_build_tt_train's).

The synthetic trees are SYNTHETIC; no value here is a measurement.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
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
TRAIN_TOOL = REPO / "tools" / "build_tt_train.py"
PACK_DIR = REPO / "assets" / "examples" / "tt-examples-v0"
SPEC = PACK_DIR / "spec.yaml"
MANIFEST = PACK_DIR / "MANIFEST.yaml"
DATASET = REPO / "assets" / "train" / "tt-train-v0" / "dataset.yaml"
REFERENCES = REPO / "assets" / "bench" / "tt-pairs-v0"
EXPECTED = ["shard_data_rm", "add_2_integers_in_riscv"]
RUN_LINES = 3


def load_tool(path: Path, name: str) -> ModuleType:
    """Import a tools/ script as the other tool tests import theirs."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, f"missing {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def tool() -> ModuleType:
    """Import tools/build_tt_examples.py."""
    return load_tool(TOOL, "build_tt_examples_under_test")


def train() -> ModuleType:
    """Import tools/build_tt_train.py, whose Spec and HeldOut the examples tool reads."""
    return load_tool(TRAIN_TOOL, "build_tt_train_for_examples")


def sha256(data: bytes) -> str:
    """Return the sha256 hex digest of `data`."""
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# The tool's pieces, on SYNTHETIC trees

ROOT = "examples"
FILES = {
    f"{ROOT}/alpha/alpha.cpp": b"int main() {\n    alpha();\n    return 0;\n}\n",
    f"{ROOT}/alpha/kernels/writer.cpp": b"void kernel_main() {\n    write_out();\n}\n",
    f"{ROOT}/alpha/kernels/reader.cpp": b"void kernel_main() {\n    read_in();\n}\n",
    f"{ROOT}/alpha/README.md": b"# SYNTHETIC\n",
    f"{ROOT}/beta/beta.cpp": b"int beta() {\n    beta_work();\n    return 1;\n}\n",
    f"{ROOT}/tier/tier.cpp": b"TIER one;\nTIER two;\nTIER three;\nTIER four;\n",
    f"{ROOT}/gone/gone.cpp": b"int gone() {\n    return 2;\n}\n",
}
ALPHA_FILES = ("alpha.cpp", "kernels/reader.cpp", "kernels/writer.cpp")


def dataset_of(**changes: Any) -> Any:
    """Return a data set Spec over the SYNTHETIC root: alpha and beta included; tier and gone excluded."""
    entry = train().Entry(task="SYNTHETIC", described_from="SYNTHETIC")
    values: dict[str, Any] = {
        "dataset": "synthetic-set", "pin": "tt-metal", "root": ROOT, "suite": "assets/bench/synthetic.yaml",
        "run_lines": RUN_LINES, "near_copy": 0.5, "prompt": "SYNTHETIC $task",
        "include": {"alpha": entry, "beta": entry},
        "exclude": {"tier": "Tier A item held", "gone": "SYNTHETIC duplicate"},
    }
    values.update(changes)
    return train().Spec(**values)


def held_of(**changes: Any) -> Any:
    """Return a HeldOut: one held item whose example directory is examples/tier, and one tracked reference."""
    values: dict[str, Any] = {
        "suite": "synthetic", "suite_path": "assets/bench/synthetic.yaml", "suite_sha256": "0" * 64,
        "items": ("held",), "tier_dirs": {f"{ROOT}/tier": "held"},
        "references": {"assets/bench/synthetic/held/tt/held.cpp": "REF one;\nREF two;\nREF three;\n"},
    }
    values.update(changes)
    return train().HeldOut(**values)


def spec_of(*examples: tuple[str, tuple[str, ...]]) -> Any:
    """Return a pack spec listing `examples` ((directory, files) pairs) in order."""
    listed = examples or (("alpha", ALPHA_FILES), ("beta", ("beta.cpp",)))
    entries = tuple(tool().ExampleSpec(directory=name, files=files) for name, files in listed)
    return tool().PackSpec(pack="syn-pack", pin="tt-metal", dataset="assets/train/synthetic.yaml", examples=entries)


def build(files: dict[str, bytes] | None = None, spec: Any = None, **held: Any) -> Any:
    """Run the tool's build on the SYNTHETIC tree."""
    return tool().build(FILES if files is None else files, spec or spec_of(), dataset_of(), held_of(**held))


def refusal(**kwargs: Any) -> str:
    """Return the ValueError message build raises for `kwargs`."""
    with pytest.raises(ValueError) as refused:
        build(**kwargs)
    return str(refused.value)


def test_build_gives_the_files_in_spec_order_with_their_bytes() -> None:
    built = build()
    assert list(built.fragments) == ["alpha/alpha.cpp", "alpha/kernels/reader.cpp", "alpha/kernels/writer.cpp",
                                     "beta/beta.cpp"]
    for path, data in built.fragments.items():
        assert data == FILES[f"{ROOT}/{path}"]
    assert [example["name"] for example in built.examples] == ["alpha", "beta"]
    assert [entry["path"] for entry in built.examples[0]["files"]] == list(built.fragments)[:3]
    host = FILES[f"{ROOT}/alpha/alpha.cpp"]
    assert built.examples[0]["files"][0] == {"path": "alpha/alpha.cpp", "sha256": sha256(host), "bytes": len(host)}


@pytest.mark.parametrize("directory", ["tier", "gone", "nowhere"])
def test_a_directory_outside_the_include_list_is_refused(directory: str) -> None:
    message = refusal(spec=spec_of(("alpha", ALPHA_FILES), (directory, (f"{directory}.cpp",))))
    assert directory in message and "include" in message, message


def test_an_unlisted_code_file_is_refused() -> None:
    message = refusal(spec=spec_of(("alpha", ("alpha.cpp", "kernels/reader.cpp"))))
    assert "kernels/writer.cpp" in message, message


@pytest.mark.parametrize("extra", ["README.md", "kernels/missing.cpp"])
def test_a_listed_file_that_is_no_code_file_of_the_directory_is_refused(extra: str) -> None:
    message = refusal(spec=spec_of(("alpha", (*ALPHA_FILES, extra))))
    assert extra in message, message


@pytest.mark.parametrize("files", [("kernels/reader.cpp", "alpha.cpp", "kernels/writer.cpp"),
                                   ("alpha.cpp", "kernels/writer.cpp", "kernels/reader.cpp")],
                         ids=["host-not-first", "kernels-unsorted"])
def test_the_host_program_first_and_sorted_kernels_are_required(files: tuple[str, ...]) -> None:
    message = refusal(spec=spec_of(("alpha", files)))
    assert "alpha" in message and "host program" in message, message


def test_a_file_with_a_tracked_reference_run_is_refused() -> None:
    files = {**FILES, f"{ROOT}/alpha/kernels/reader.cpp": b"x;\n  REF one;\nREF two;\n\nREF   three;\ny;\n"}
    message = refusal(files=files)
    assert "alpha/kernels/reader.cpp" in message and "assets/bench/synthetic/held/tt/held.cpp" in message, message


def test_a_copy_of_a_tier_a_file_is_refused() -> None:
    message = refusal(files={**FILES, f"{ROOT}/alpha/kernels/reader.cpp": FILES[f"{ROOT}/tier/tier.cpp"]})
    assert "alpha" in message and f"{ROOT}/tier/tier.cpp" in message, message


def test_a_near_copy_of_a_tier_a_file_is_refused() -> None:
    near = b"TIER one;\nTIER two;\nTIER three;\nTIER four;\nmine;\n"
    assert "near copy" in refusal(files={**FILES, f"{ROOT}/alpha/kernels/reader.cpp": near})


def test_a_held_directory_missing_from_the_tree_is_refused() -> None:
    files = {path: data for path, data in FILES.items() if not path.startswith(f"{ROOT}/tier/")}
    message = refusal(files=files)
    assert f"{ROOT}/tier" in message and "Agent Rule 5" in message, message


# ---------------------------------------------------------------------------
# fetch, on a SYNTHETIC GitHub


COMMIT = "c" * 40
PIN = {"NAME": "tt-metal", "URL": "https://github.com/syn/tt-metal.git", "COMMIT": COMMIT, "VERSION": "cccccccc"}


def git_blob(data: bytes) -> str:
    """Return the git blob id of `data`."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def fake_github(files: dict[str, bytes], fetched: list[str], *, corrupt: str | None = None) -> Any:
    """Return an opener serving the SYNTHETIC tree: the commit's top tree, the root's recursive tree, and raw files."""
    api = "https://api.github.com/repos/syn/tt-metal/git/trees/"
    raw = f"https://raw.githubusercontent.com/syn/tt-metal/{COMMIT}/"
    listing = [{"path": path[len(ROOT) + 1:], "type": "blob", "mode": "100644", "sha": git_blob(data)}
               for path, data in files.items()]

    def opener(url: str) -> bytes:
        if url == api + COMMIT:
            return json.dumps({"tree": [{"path": ROOT, "type": "tree", "sha": "t" * 40}]}).encode()
        if url == api + "t" * 40 + "?recursive=1":
            return json.dumps({"tree": listing}).encode()
        assert url.startswith(raw), url
        path = url[len(raw):]
        fetched.append(path)
        return b"SYNTHETIC corrupt\n" if path == corrupt else files[path]

    return opener


def test_fetch_downloads_the_listed_and_held_directories_only(tmp_path: Path) -> None:
    fetched: list[str] = []
    count = tool().fetch(tmp_path / "tree", spec_of(), dataset_of(), held_of(), PIN, fake_github(FILES, fetched))
    expected = sorted(path for path in FILES if not path.startswith(f"{ROOT}/gone/"))
    assert sorted(fetched) == expected and count == len(expected)
    for path in expected:
        assert (tmp_path / "tree" / path).read_bytes() == FILES[path]
    record = (tmp_path / "tree" / "lassi-fetch.txt").read_bytes().decode("ascii").splitlines()
    assert record[0] == COMMIT and record[1].startswith("pack syn-pack:"), record
    tool().check_tree(tmp_path / "tree", PIN, "syn-pack")
    with pytest.raises(ValueError, match="syn-other"):
        tool().check_tree(tmp_path / "tree", PIN, "syn-other")


def test_fetch_refuses_a_file_that_is_not_its_blob(tmp_path: Path) -> None:
    opener = fake_github(FILES, [], corrupt=f"{ROOT}/alpha/alpha.cpp")
    with pytest.raises(ValueError, match="alpha.cpp"):
        tool().fetch(tmp_path / "tree", spec_of(), dataset_of(), held_of(), PIN, opener)


def test_fetch_refuses_a_destination_inside_the_repository() -> None:
    with pytest.raises(ValueError, match="outside the repository"):
        tool().fetch(REPO / "assets" / "examples" / "x", spec_of(), dataset_of(), held_of(), PIN,
                     fake_github(FILES, []))


# ---------------------------------------------------------------------------
# The manifest, check_pack, and load_pack


def written_pack(tmp_path: Path) -> Path:
    """Build the SYNTHETIC pack and write it, fragments and manifest, under tmp_path/syn-pack; return the tree."""
    built = build()
    document = tool().manifest_document(built, spec_of(), PIN, spec_sha256="1" * 64, spec_name="spec.yaml",
                                        checks={"dataset": "assets/train/synthetic.yaml"})
    tree = tmp_path / "syn-pack"
    tool().write_pack(tree, built.fragments, document)
    return tree


def test_the_manifest_records_every_file_and_the_pack_sha256(tmp_path: Path) -> None:
    tree = written_pack(tmp_path)
    data = (tree / "MANIFEST.yaml").read_bytes()
    assert data.isascii() and b"\r" not in data
    document = yaml.safe_load(data.decode("ascii"))
    assert document["generator"] == "tools/build_tt_examples.py" and document["pack"] == "syn-pack"
    assert (document["repo"], document["commit"]) == (PIN["URL"], COMMIT)
    rows = [entry for example in document["examples"] for entry in example["files"]]
    assert [row["path"] for row in rows] == list(build().fragments)
    for row in rows:
        assert row["sha256"] == sha256((tree / row["path"]).read_bytes())
        assert row["bytes"] == len((tree / row["path"]).read_bytes())
    lines = "".join(f"{row['sha256']}  {row['path']}\n" for row in rows)
    assert document["sha256"] == sha256(lines.encode("ascii"))
    assert examples_module.check_pack(tree) == []


def test_load_pack_returns_the_examples_in_order(tmp_path: Path) -> None:
    examples = examples_module.load_pack(written_pack(tmp_path))
    assert [example.name for example in examples] == ["alpha", "beta"]
    assert [path for path, _ in examples[0].files] == list(ALPHA_FILES)
    assert examples[0].files[0][1] == FILES[f"{ROOT}/alpha/alpha.cpp"].decode("ascii")


@pytest.mark.parametrize("change", ["missing", "same-size", "longer", "pack-sha256"])
def test_check_pack_names_each_difference(tmp_path: Path, change: str) -> None:
    tree = written_pack(tmp_path)
    target = tree / "alpha" / "kernels" / "reader.cpp"
    if change == "missing":
        target.unlink()
    elif change == "same-size":
        target.write_bytes(target.read_bytes().replace(b"read_in", b"READ_IN"))
    elif change == "longer":
        target.write_bytes(target.read_bytes() + b"// more\n")
    else:
        manifest = tree / "MANIFEST.yaml"
        document = yaml.safe_load(manifest.read_bytes().decode("ascii"))
        document["sha256"] = "f" * 64
        manifest.write_bytes(yaml.safe_dump(document, sort_keys=False).encode("ascii"))
    problems = examples_module.check_pack(tree)
    assert problems, "a difference is found"
    named = "sha256" if change == "pack-sha256" else "reader.cpp"
    assert any(named in problem for problem in problems), problems
    with pytest.raises(ValueError, match="tools/build_tt_examples.py"):
        examples_module.load_pack(tree)


# ---------------------------------------------------------------------------
# The tracked spec, manifest, and ignore lines


def spec_data() -> dict[str, Any]:
    """Return the tracked spec as YAML data."""
    return yaml.safe_load(SPEC.read_bytes().decode("ascii"))


def manifest_data() -> dict[str, Any]:
    """Return the tracked manifest."""
    assert MANIFEST.is_file(), f"{MANIFEST} is missing; run uv run tools/build_tt_examples.py build --tree <tree>"
    return yaml.safe_load(MANIFEST.read_bytes().decode("ascii"))


def test_the_tracked_spec_lists_the_two_examples_from_the_include_list() -> None:
    data = spec_data()
    dataset = yaml.safe_load(DATASET.read_bytes().decode("ascii"))
    assert data["pack"] == "tt-examples-v0" and data["pin"] == dataset["pin"]
    assert data["dataset"] == DATASET.relative_to(REPO).as_posix()
    assert [entry["directory"] for entry in data["examples"]] == EXPECTED
    for entry in data["examples"]:
        assert entry["directory"] in dataset["include"] and entry["directory"] not in dataset["exclude"]
        host, *kernels = entry["files"]
        assert host.endswith(".cpp") and not host.startswith("kernels/"), entry
        assert kernels and kernels == sorted(kernels) and all(path.startswith("kernels/") for path in kernels)
    loaded = tool().load_spec(SPEC)
    assert [example.directory for example in loaded.examples] == EXPECTED


def test_the_tracked_manifest_matches_the_spec_and_the_pin() -> None:
    found = manifest_data()
    pin = pins.read_pin("tt-metal")
    assert found["generator"] == "tools/build_tt_examples.py" and found["pack"] == "tt-examples-v0"
    assert (found["repo"], found["commit"]) == (pin["URL"], pin["COMMIT"])
    assert found["spec_sha256"] == sha256(SPEC.read_bytes()), "rebuild: the spec changed"
    listed = [[f"{entry['directory']}/{path}" for path in entry["files"]] for entry in spec_data()["examples"]]
    assert [[row["path"] for row in example["files"]] for example in found["examples"]] == listed
    assert [example["name"] for example in found["examples"]] == EXPECTED
    rows = [row for example in found["examples"] for row in example["files"]]
    assert all(len(row["sha256"]) == 64 and row["bytes"] > 0 for row in rows)
    lines = "".join(f"{row['sha256']}  {row['path']}\n" for row in rows)
    assert found["sha256"] == sha256(lines.encode("ascii"))


def test_only_the_spec_and_the_manifest_are_tracked() -> None:
    lines = (REPO / ".gitignore").read_bytes().decode("ascii").splitlines()
    for line in ("/assets/examples/tt-examples-v0/*", "!/assets/examples/tt-examples-v0/MANIFEST.yaml",
                 "!/assets/examples/tt-examples-v0/spec.yaml"):
        assert line in lines, f".gitignore lacks {line}"


def own_runs(text: str, size: int = RUN_LINES) -> set[tuple[str, ...]]:
    """Return every run of `size` consecutive non-blank lines of `text`, each stripped and whitespace-collapsed."""
    lines = [" ".join(line.split()) for line in text.split("\n") if line.strip()]
    return {tuple(lines[index:index + size]) for index in range(len(lines) - size + 1)}


def built_fragments() -> list[Path]:
    """Return the generated fragments the manifest lists, or skip naming the build command when any is absent."""
    paths = [PACK_DIR / row["path"] for example in manifest_data()["examples"] for row in example["files"]]
    if not all(path.is_file() for path in paths):
        pytest.skip("the tt-examples-v0 fragments are not built; build them with "
                    "uv run tools/build_tt_examples.py fetch --dest <dir>, then build --tree <dir>")
    return paths


def test_the_built_fragments_match_the_manifest_and_hold_no_tier_a_run() -> None:
    paths = built_fragments()
    assert examples_module.check_pack(PACK_DIR) == []
    references: dict[tuple[str, ...], str] = {}
    for path in sorted(REFERENCES.rglob("*")):
        if path.is_file():
            for run in own_runs(path.read_bytes().decode("utf-8")):
                references.setdefault(run, path.relative_to(REPO).as_posix())
    assert references, "no Tier A reference was read"
    for path in paths:
        shared = sorted(own_runs(path.read_bytes().decode("utf-8")) & set(references))
        assert not shared, f"{path.name} holds {shared[0]} of {references[shared[0]]}"
    examples = examples_module.load_pack(PACK_DIR)
    assert [example.name for example in examples] == EXPECTED
    for example in examples:
        host, *kernels = [path for path, _ in example.files]
        assert host.endswith(".cpp") and "int main" in example.files[0][1] and kernels, example.name


def test_check_mode_passes_on_the_built_pack() -> None:
    built_fragments()
    assert tool().main(["check"]) == 0
