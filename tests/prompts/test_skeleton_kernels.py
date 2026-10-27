"""Tests for kernel skeletons beside a host program skeleton in lassi.prompts.skeletons (task P17.39).

Bible: Project Recipes (Notes, the Skeleton bullet), Agent Rules 5 and 10.
The owner's answer to OQ-096 (a), "Item-agnostic only", governs every
skeleton; task P17.39 lets a skeleton for the C++ -> TT kernels arm carry
item-agnostic kernel skeletons too.

What they fix (SYNTHETIC skeletons written here):

- A manifest's files are host.cpp and block.txt, then zero or more kernel
  skeletons, each a .cpp path with a directory named kernels in it, in
  order; Skeleton.kernels holds them as (path, text) in that order, and a
  skeleton without them reads exactly as before.
- A skeleton with kernels needs a block that uses $kernels, which renders
  them as FILE blocks in manifest order; a block that uses $kernels with no
  kernel listed is refused, and so is any other placeholder.
- A listed kernel that is missing, changed, or not under kernels/ is
  refused, naming it; the manifest's sha256 covers every listed file.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from lassi.prompts import skeletons

COMMIT = "5280a9cfb00998fd49667a29523d03aee905c129"
HOST = "// SYNTHETIC host skeleton\nint main() { return 0; }\n"
KERNELS = {
    "kernels/dataflow/reader.cpp": "// SYNTHETIC reader skeleton\nvoid kernel_main() {}\n",
    "kernels/compute/compute.cpp": "// SYNTHETIC compute skeleton\nvoid kernel_main() {}\n",
}
BLOCK = "SYNTHETIC block.\n```cpp\n$skeleton```\nSYNTHETIC kernels:\n\n$kernels"
HOST_BLOCK = "SYNTHETIC block.\n```cpp\n$skeleton```\n"


def write(tree: Path, files: dict[str, str], name: str = "syn-skeleton") -> Path:
    """Write the files and a manifest naming them, in order, under `tree`/<name>; return the skeleton's directory."""
    directory = tree / name
    entries = []
    for path, text in files.items():
        (directory / path).parent.mkdir(parents=True, exist_ok=True)
        (directory / path).write_bytes(text.encode("ascii"))
        data = text.encode("ascii")
        entries.append({"path": path, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
    document = {"skeleton": name, "language": "ttkh", "pin": "tt-metal", "commit": COMMIT, "files": entries,
                "sha256": skeletons.files_sha256([(entry["path"], entry["sha256"]) for entry in entries])}
    (directory / skeletons.MANIFEST).write_bytes(yaml.safe_dump(document, sort_keys=False).encode("ascii"))
    return directory


def test_a_skeleton_with_kernels_reads_them_in_order_and_renders_them(tmp_path: Path) -> None:
    tree = write(tmp_path, {"host.cpp": HOST, "block.txt": BLOCK, **KERNELS})
    loaded = skeletons.load_skeleton(tree)
    assert loaded.kernels == tuple(KERNELS.items())
    text = skeletons.block_text(loaded)
    assert HOST in text and "$kernels" not in text
    for path, body in KERNELS.items():
        assert f"// FILE: {path}\n{body}" in text
    assert text.index("reader.cpp") < text.index("compute.cpp"), "the manifest's order"


def test_a_skeleton_without_kernels_reads_as_before(tmp_path: Path) -> None:
    loaded = skeletons.load_skeleton(write(tmp_path, {"host.cpp": HOST, "block.txt": HOST_BLOCK}))
    assert loaded.kernels == ()
    assert skeletons.block_text(loaded) == HOST_BLOCK.replace("$skeleton", HOST)


@pytest.mark.parametrize(("files", "words"), [
    ({"host.cpp": HOST, "block.txt": HOST_BLOCK, **KERNELS}, "$kernels"),
    ({"host.cpp": HOST, "block.txt": BLOCK}, "no kernel"),
    ({"host.cpp": HOST, "block.txt": BLOCK + "$other", **KERNELS}, "$other"),
], ids=["kernels-unshown", "kernels-field-with-none", "other-field"])
def test_a_block_that_does_not_fit_its_kernels_is_refused(tmp_path: Path, files: dict[str, str], words: str) -> None:
    with pytest.raises(ValueError, match=words.replace("$", r"\$")):
        skeletons.load_skeleton(write(tmp_path, files))


@pytest.mark.parametrize("path", ["reader.cpp", "kernels/reader.h", "kernels/../x.cpp", "src/kernels"])
def test_a_kernel_path_outside_kernels_or_not_cpp_is_refused(tmp_path: Path, path: str) -> None:
    tree = write(tmp_path, {"host.cpp": HOST, "block.txt": BLOCK, "kernels/ok.cpp": "// SYNTHETIC\n"})
    document = yaml.safe_load((tree / skeletons.MANIFEST).read_bytes().decode("ascii"))
    document["files"][2]["path"] = path
    (tree / skeletons.MANIFEST).write_bytes(yaml.safe_dump(document, sort_keys=False).encode("ascii"))
    with pytest.raises(ValueError, match="kernels"):
        skeletons.load_skeleton(tree)


@pytest.mark.parametrize("change", ["missing", "changed"])
def test_a_listed_kernel_off_its_manifest_is_refused(tmp_path: Path, change: str) -> None:
    tree = write(tmp_path, {"host.cpp": HOST, "block.txt": BLOCK, **KERNELS})
    target = tree / "kernels" / "compute" / "compute.cpp"
    if change == "missing":
        target.unlink()
    else:
        target.write_bytes(target.read_bytes().replace(b"SYNTHETIC", b"SYNTHETIX"))
    with pytest.raises(ValueError, match="compute.cpp"):
        skeletons.load_skeleton(tree)


def test_the_manifest_sha256_covers_every_kernel(tmp_path: Path) -> None:
    tree = write(tmp_path, {"host.cpp": HOST, "block.txt": BLOCK, **KERNELS})
    without = write(tmp_path, {"host.cpp": HOST, "block.txt": HOST_BLOCK}, "syn-host-only")
    assert skeletons.load_skeleton(tree).sha256 != skeletons.load_skeleton(without).sha256


def test_the_request_note_counts_the_kernel_skeletons(tmp_path: Path) -> None:
    from lassi.core.stages import skeleton_note

    with_kernels = skeletons.load_skeleton(write(tmp_path, {"host.cpp": HOST, "block.txt": BLOCK, **KERNELS}))
    host_only = skeletons.load_skeleton(write(tmp_path, {"host.cpp": HOST, "block.txt": HOST_BLOCK}, "syn-host"))
    assert skeleton_note(with_kernels).message.endswith(f"ttkh host program, and {len(KERNELS)} kernel skeletons)")
    assert skeleton_note(host_only).message.endswith("characters of ttkh host program)"), "as before"
