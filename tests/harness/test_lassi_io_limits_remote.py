"""Remote tests of lassi_io.h's reader at its limits, on the build host (task P4.3).

Bible: Harness Contract (lassi_io.h refusals), Sandbox, Agent Rules 6 and 7.

The contract these tests fix, in assets/harness/c/lassi_io.h:

- lassi_io_read zeroes *out first whenever out is not NULL, so the array is
  all zero bytes after any refusal: a NULL path, a missing file, a bad
  magic, and a rank of 224 or 60000 with every dim 2**64 - 1 (a byte count
  past 64 bits). Each refusal prints one line to stderr naming the file, or
  "(no path)" for a NULL path.
- A zero dim among dims of 2**64 - 1 reads as an empty array.

One C99 program written here, probe.c, not generated code, fills an array
with 0xAB bytes before each read and prints one line per read: "ok <dtype
code> <rank> <count> <nbytes>", "refused zeroed", or "refused not-zeroed".
Its first read is of a NULL path, then one per argument. It is compiled and
run as test_lassi_io_remote.py compiles and runs its programs, with that
module's helpers, loaded by its path: gcc with -std=c99 -O2 -Wall -Wextra
-pedantic -Werror in the compile sandbox, then NativeExecutor.run, so only in
the sandbox (Agent Rule 6). The skip rule and LASSI_REQUIRE_SANDBOX=1 are that
module's. Every file lies in a temp directory under $LASSI_RUNS_ROOT that the
test removes (Agent Rule 7). Run it with `uv run tools/rx.py run --
'LASSI_REQUIRE_SANDBOX=1 uv run pytest -q -m remote tests/harness/test_lassi_io_limits_remote.py'`.
No value here is a measurement.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import struct
import tempfile
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

MAGIC = b"LASSIIO\x00"
MAX_U64 = 2**64 - 1

PROBE_SOURCE = r"""/* A probe of lassi_io_read: is the array zeroed after each refusal? */
#include <stdio.h>
#include <string.h>

#include "lassi_io.h"

static int is_zero(const lassi_io_array *array) {
    const unsigned char *bytes = (const unsigned char *)array;
    size_t i;
    for (i = 0; i < sizeof *array; ++i) {
        if (bytes[i] != 0) {
            return 0;
        }
    }
    return 1;
}

static void probe(const char *path) {
    lassi_io_array array;
    memset(&array, 0xAB, sizeof array);
    if (lassi_io_read(path, &array) == 0) {
        printf("ok %u %u %llu %llu\n", (unsigned)array.dtype, (unsigned)array.rank,
               (unsigned long long)array.count, (unsigned long long)array.nbytes);
        lassi_io_free(&array);
    } else {
        printf("refused %s\n", is_zero(&array) ? "zeroed" : "not-zeroed");
    }
    fflush(stdout);
}

int main(int argc, char **argv) {
    int i;
    probe(NULL);
    for (i = 1; i < argc; ++i) {
        probe(argv[i]);
    }
    return 0;
}
"""


def helpers() -> ModuleType:
    """Load test_lassi_io_remote.py by its path, for compile_program, run, C_FLAGS, PROBLEM, and REQUIRE."""
    path = Path(__file__).with_name("test_lassi_io_remote.py")
    spec = importlib.util.spec_from_file_location("lassi_io_remote_helpers", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REMOTE = helpers()
pytestmark = [
    pytest.mark.remote,
    pytest.mark.skipif(bool(REMOTE.PROBLEM) and not REMOTE.REQUIRE, reason=REMOTE.PROBLEM or "the host can run it"),
]


def build(dims: tuple[int, ...], data: bytes, code: int = 1, name: bytes = b"x") -> bytes:
    """Build a file by hand: magic, version 1, dtype code, rank, dims, name, zero padding, data."""
    head = MAGIC + struct.pack("<III", 1, code, len(dims)) + struct.pack(f"<{len(dims)}Q", *dims)
    head += struct.pack("<I", len(name)) + name
    return head + bytes(-len(head) % 8) + data


@pytest.fixture
def base() -> Iterator[Path]:
    """Create a fresh directory under $LASSI_RUNS_ROOT and remove it afterwards; fail when the host cannot run it."""
    if REMOTE.PROBLEM:
        pytest.fail(f"LASSI_REQUIRE_SANDBOX=1, but {REMOTE.PROBLEM}")
    root = Path(tempfile.mkdtemp(prefix="lassi-io-limits.", dir=os.environ["LASSI_RUNS_ROOT"]))
    try:
        yield root
    finally:
        shutil.rmtree(root)


def test_the_c_reader_zeroes_the_array_after_every_refusal(base: Path) -> None:
    compiled = REMOTE.compile_program(base / "build", "probe.c", PROBE_SOURCE, "gcc", REMOTE.C_FLAGS)
    assert compiled.returncode == 0, f"status {compiled.returncode}:\n{compiled.stderr}"
    workdir = base / "attempt00" / "build"
    workdir.mkdir(parents=True)
    program = workdir / "main"
    shutil.copy2(base / "build" / "main", program)
    valid = build((2,), struct.pack("<2f", 1.0, -2.0))
    files = {
        "valid.lassiio": (valid, "ok 1 1 2 8"),
        "rank224.lassiio": (build((MAX_U64,) * 224, b"\x00" * 8), "refused zeroed"),
        "rank60000.lassiio": (build((MAX_U64,) * 60000, b"\x00" * 8), "refused zeroed"),
        "empty.lassiio": (build((MAX_U64,) * 223 + (0,), b"", code=7), "ok 7 224 0 0"),
        "bad_magic.lassiio": (b"X" + valid[1:], "refused zeroed"),
    }
    for file, (data, _) in files.items():
        (workdir / file).write_bytes(data)
    result = REMOTE.run(program, [*files, "missing.lassiio"])
    assert result.exit_code == 0, result
    expected = ["refused zeroed", *(line for _, line in files.values()), "refused zeroed"]
    assert result.stdout.splitlines() == expected
    unnamed = [
        file
        for file in ("rank224.lassiio", "rank60000.lassiio", "bad_magic.lassiio", "missing.lassiio")
        if file not in result.stderr
    ]
    assert not unnamed, f"refusals whose stderr does not name the file: {unnamed}\n{result.stderr[-2000:]}"
    assert "(no path)" in result.stderr
