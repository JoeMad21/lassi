"""Tests for task P17.41's compute-only C units and the suite manifest's `compute` key.

Bible: Benchmark Suites (tt-pairs-v0-ttkernel; manifest keys), Language
Frontends (C through cgeist to affine, scf, and memref), Agent Rule 5.
plans/spikes/mlir-issues.md, Half 2: cgeist (Polygeist 77c04bb2) aborted on
all five items' whole sources, most likely on the I/O harness lassi_io.h
(inferred), and three items keep their computation inside main, so no
function could be lowered alone. The fix: per item, a tracked compute-only C
translation unit, the item's computation as one function with C linkage (an
extern "C" block when compiled as C++), with no I/O, no lassi_io.h, and no
main, which cgeist lowers by its C name.

What they fix:

- An item may declare `compute: {tracked: <path under assets/>, function:
  <C identifier>, sha256: <64 hex>}`; SuiteItem.compute holds it and
  Suite.compute_source reads it (refusing bytes that differ from the sha256,
  an item without one, and an item refused to the purpose). A malformed key
  is refused at load naming it.
- Every tt-pairs-v0-ttkernel item declares one, and the unit holds no I/O
  and exactly one function with external linkage, the one named.
- Where a C compiler is on PATH, each unit computes what the C++ reference
  computes on SYNTHETIC inputs whose sums and products are exact, so the
  result does not depend on rounding or contraction (skipped otherwise; no
  compiler runs on the Windows host).
"""

from __future__ import annotations

import hashlib
import re
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from lassi.bench import load_suite, registry

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "assets" / "bench" / "tt-pairs-v0-ttkernel.yaml"
SUITE = load_suite(MANIFEST)
ITEMS = ("eltwise_binary", "eltwise_sfpu", "loopback", "matmul_multi_core", "matmul_single_core")
# What a compute unit may not hold: the I/O harness, a program entry, or file and stream I/O.
FORBIDDEN = ("lassi_io", "main(", "printf", "fopen", "FILE", "stdio", "std::")
# A function definition with external linkage at file scope: a type, the name, and its parameters, then `{`.
DEFINITION = re.compile(r"^(?!static\b)[A-Za-z_][\w ]*?\b(?P<name>[A-Za-z_]\w*)\s*\([^;{]*\)\s*\{", re.MULTILINE)


def unit(item: str) -> tuple[registry.ComputeUnit, str]:
    """Return an item's compute unit and its text."""
    found = SUITE.items[item].compute
    assert found is not None, f"{item} declares no compute unit"
    return found, (REPO / found.path).read_text(encoding="ascii")


@pytest.mark.parametrize("item", ITEMS)
def test_every_ttkernel_item_declares_a_tracked_compute_unit_with_its_sha256(item: str) -> None:
    found, _ = unit(item)
    assert found.path == f"assets/bench/tt-pairs-v0-ttkernel/{item}/{item}.c"
    assert found.function == f"lassi_{item}"
    assert found.sha256 == hashlib.sha256((REPO / found.path).read_bytes()).hexdigest()
    source = SUITE.compute_source(item, purpose="eval")
    assert (source.path, source.file, source.function) == (found.path, f"{item}.c", found.function)


@pytest.mark.parametrize("item", ITEMS)
def test_a_compute_unit_is_compute_only_with_one_external_function(item: str) -> None:
    found, text = unit(item)
    data = (REPO / found.path).read_bytes()
    assert data.isascii() and b"\r" not in data
    code = re.sub(r"//[^\n]*", "", re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL))
    for word in FORBIDDEN:
        assert word not in code, f"{item}: the unit's code holds {word!r}"
    assert [match.group("name") for match in DEFINITION.finditer(code)] == [found.function]
    assert 'extern "C" {' in text and "#ifdef __cplusplus" in text, "C linkage when compiled as C++"
    includes = re.findall(r"#include\s*<([^>]+)>", text)
    assert set(includes) <= {"math.h"}, includes


def test_the_compute_key_is_read_and_checked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    copy = tmp_path / "repo"
    found, _ = unit("loopback")
    (copy / found.path).parent.mkdir(parents=True)
    (copy / found.path).write_bytes((REPO / found.path).read_bytes())
    monkeypatch.setattr(registry, "TRACKED_ROOT", copy)
    assert SUITE.compute_source("loopback", purpose="eval").text.startswith("/*")
    (copy / found.path).write_bytes(b"void lassi_loopback(void) {}\n")
    with pytest.raises(ValueError, match="sha256"):
        SUITE.compute_source("loopback", purpose="eval")
    assert SUITE.items["loopback"].split == "unassigned"
    with pytest.raises(registry.EvalSplitError):
        SUITE.compute_source("loopback", purpose="train")


def write_manifest(tmp_path: Path, change: Any) -> Path:
    """Write a copy of the manifest with `change` applied to loopback's compute entry; return its path."""
    data = yaml.safe_load(MANIFEST.read_bytes().decode("utf-8"))
    change(data["items"]["loopback"])
    path = tmp_path / MANIFEST.name
    path.write_bytes(yaml.safe_dump(data).encode("utf-8"))
    return path


@pytest.mark.parametrize(
    "change",
    [
        lambda item: item["compute"].pop("function"),
        lambda item: item["compute"].update(function="2bad"),
        lambda item: item["compute"].update(tracked="/abs/loopback.c"),
        lambda item: item["compute"].update(tracked="src/loopback.c"),
        lambda item: item["compute"].update(sha256="xyz"),
        lambda item: item["compute"].update(extra=1),
        lambda item: item.update(compute="loopback.c"),
    ],
    ids=["no-function", "bad-function", "absolute", "not-assets", "bad-sha", "extra-key", "not-a-mapping"],
)
def test_a_malformed_compute_key_is_refused(tmp_path: Path, change: Any) -> None:
    with pytest.raises(ValueError, match="compute"):
        load_suite(write_manifest(tmp_path, change))


def test_an_item_without_a_compute_unit_is_refused_by_compute_source(tmp_path: Path) -> None:
    suite = load_suite(write_manifest(tmp_path, lambda item: item.pop("compute")))
    assert suite.items["loopback"].compute is None
    with pytest.raises(ValueError, match="no compute unit"):
        suite.compute_source("loopback", purpose="eval")


# ---------------------------------------------------------------------------
# The arithmetic, where a C compiler is on PATH


def compiler() -> str | None:
    """Return a C compiler on PATH, or None."""
    return next((found for name in ("cc", "gcc", "clang") if (found := shutil.which(name))), None)


def f32(value: float) -> float:
    """Return `value` rounded to float32."""
    return float(struct.unpack("<f", struct.pack("<f", value))[0])


# SYNTHETIC drivers: each calls the unit's function on small inputs, all multiples of 1/8 below 4, so every sum
# and product is exact in float32, and prints each result as a decimal float.
DRIVERS = {
    "eltwise_binary": (
        "void lassi_eltwise_binary(const float*, const float*, float*, long);\n"
        "int main(void) { float a[8], b[8], c[8]; for (int i = 0; i < 8; ++i) { a[i] = i / 8.0f; b[i] = -i / 4.0f; }\n"
        "  lassi_eltwise_binary(a, b, c, 8); for (int i = 0; i < 8; ++i) printf(\"%.9g\\n\", c[i]); return 0; }\n"
    ),
    "matmul_single_core": (
        "void lassi_matmul_single_core(const float*, const float*, float*, long, long, long);\n"
        "int main(void) { float a[6], b[12], c[8]; for (int i = 0; i < 6; ++i) a[i] = (i % 5) / 8.0f;\n"
        "  for (int i = 0; i < 12; ++i) b[i] = (i % 7) / 8.0f; lassi_matmul_single_core(a, b, c, 2, 3, 4);\n"
        "  for (int i = 0; i < 8; ++i) printf(\"%.9g\\n\", c[i]); return 0; }\n"
    ),
    "matmul_multi_core": (
        "void lassi_matmul_multi_core(const float*, const float*, float*, long, long, long);\n"
        "int main(void) { float a[6], b[12], c[8]; for (int i = 0; i < 6; ++i) a[i] = (i % 5) / 8.0f;\n"
        "  for (int i = 0; i < 12; ++i) b[i] = (i % 7) / 8.0f; lassi_matmul_multi_core(a, b, c, 2, 3, 4);\n"
        "  for (int i = 0; i < 8; ++i) printf(\"%.9g\\n\", c[i]); return 0; }\n"
    ),
    "loopback": (
        "void lassi_loopback(const unsigned short*, unsigned short*, long, unsigned short*);\n"
        "int main(void) { static unsigned short s[2048], d[2048], t[1024];\n"
        "  for (int i = 0; i < 2048; ++i) s[i] = (unsigned short)(i * 7 + 3); lassi_loopback(s, d, 2, t);\n"
        "  for (int i = 0; i < 2048; i += 511) printf(\"%d\\n\", d[i]); return 0; }\n"
    ),
}


def expected(item: str) -> list[str]:
    """Return what the C++ reference computes for the driver's inputs, printed as the driver prints it."""
    if item == "eltwise_binary":
        return [f"{f32(i / 8 - i / 4):.9g}" for i in range(8)]
    if item == "loopback":
        return [str(i * 7 + 3) for i in range(0, 2048, 511)]
    a = [(i % 5) / 8 for i in range(6)]
    b = [(i % 7) / 8 for i in range(12)]
    out = []
    for i in range(2):
        for j in range(4):
            total = 0.0
            for r in range(3):
                total = f32(total + f32(a[i * 3 + r] * b[r * 4 + j]))
            out.append(f"{total:.9g}")
    return out


@pytest.mark.skipif(compiler() is None, reason="no C compiler on PATH (the Windows host has none)")
@pytest.mark.parametrize("item", sorted(DRIVERS))
def test_a_compute_unit_computes_what_the_reference_computes(tmp_path: Path, item: str) -> None:
    found, _ = unit(item)
    driver = tmp_path / "driver.c"
    driver.write_text("#include <stdio.h>\n" + DRIVERS[item], encoding="ascii")
    program = tmp_path / "check"
    built = subprocess.run([str(compiler()), "-O0", "-ffp-contract=off", "-o", str(program), str(REPO / found.path),
                            str(driver), "-lm"], capture_output=True, text=True, timeout=120, check=False)
    assert built.returncode == 0, built.stderr
    ran = subprocess.run([str(program)], capture_output=True, text=True, timeout=60, check=False)
    assert ran.returncode == 0 and ran.stdout.split() == expected(item)
