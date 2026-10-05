"""Tests for the CPU -> TT host-compute guard's reader, lassi.toolchains.ttmetal_guard (task P4.12).

Bible: Harness Contract (the CPU -> TT guard bullet: "must create at least
one compute kernel", "host loops that write output elements are flagged",
"data-movement-only solutions are tagged, not failed"; a C++ file with a
kernels directory in its path is a kernel source), Result Record
(Attempt.guards: None means not checked), Reward Function (a guard violation
sets R = -1), Design Principle 8, Agent Rule 3. Plan: plans/p4-ttsim.md,
P4.12. The reader's rules, readings, and limits are stated in the module
docstrings of lassi.toolchains.ttmetal_guard, lassi.toolchains._cxx_flow,
and lassi.toolchains._cxx_scan; the sections below name the part each
group of tests fixes.

The contract these tests fix:

- `read_host_compute(files, harness) -> HostComputeReading`
  (lassi.core.capabilities), where `files` are the model's files and
  `harness` the item's support files, each build-directory path -> text.
  It reads only those arguments, starts no process, and opens no file.
- Host text: every model file whose suffix is one of HOST_SUFFIXES
  (.c .cc .cpp .cxx .h .hh .hpp .hxx .inc .inl .ipp) and that is not a
  kernel source (is_kernel_source), plus the closure of what they
  `#include`: a quoted name is looked up beside the including file first,
  then at the build-directory root; an angle name only at the root; at each
  candidate the model's files are checked before the harness files; a hit is
  read whatever its suffix or directory (kernels/ included). Harness files
  and kernel sources that nothing includes are never read, and the reading
  does not depend on the dict order of either argument.
- KERNEL_CREATORS = {CreateKernel, CreateKernelFromString, KernelDescriptor,
  ProgramDescriptor}; COMPUTE_CONFIGS = {ComputeConfig,
  ComputeConfigDescriptor}; DEVICE_READS = {EnqueueReadMeshBuffer,
  ReadShard, ReadFromBuffer, ReadFromDeviceL1, ReadFromDeviceDRAMChannel,
  ReadRegFromDevice}. A name counts in code or directive lines other than
  `#include` lines, never in a comment, a literal, or an `#if 0` or
  `#if false` region.
- The reading (ttmetal_guard's module docstring, The reading): unreadable
  host text gives None and one `guard-not-checked` note; no output write
  gives None, the tag when it applies, and a `guard-not-checked` note; an
  output write with no kernel creator (rule K) gives True and one
  `guard-host-compute` note at the first sink, with no tag; a flow from an
  input read to an output write that no device read cut (rule H) gives
  True, the tag when it applies, and one `guard-host-compute` note at the
  sink; anything else gives False and the tag when it applies. The tag
  (`guard-data-movement-only`, a warning) is there exactly when a kernel
  creator is named and no compute config is.
- Every Diagnostic is parse-stage, a warning (the tag) or a note, with a
  `guard-` code; its message is one of the texts below (ttmetal_guard's
  TAG_MESSAGE, NO_KERNEL_MESSAGE, NO_OUTPUT_MESSAGE, and the rule H and
  not-checked forms), plain ASCII, a non-ASCII file name written with
  backslash escapes. The tag sits at the first kernel-creator token in a
  model file; the rule K note at the first sink; the rule H note at the
  witness's sink, naming the input read and the output write as
  <file>:<line>; the not-checked notes have no place. A diagnostic's column
  is not part of the contract, so it is not asserted.

The acceptance programs are the fixtures under fixtures/host_compute/; the
rule cases are inline strings built here. Every program is SYNTHETIC, hand
written, and uses only tt-metal API names; none is upstream text, and none
is compiled or run here (test_ttmetal_guard_remote.py builds the fixtures).
The limits ttmetal_guard's module docstring states (Limits), except the
misparses, are pinned as documented-limit tests (test_limit_*), so a change
to them is visible. No value in this module is a measurement.
"""

from __future__ import annotations

import builtins
import importlib
import inspect
import io
import os
import pathlib
import random
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Any

import pytest

from lassi.core import capabilities as capabilities_module
from lassi.core.record import Diagnostic

REPO = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "host_compute"
HARNESS_HEADER = REPO / "assets" / "harness" / "c" / "lassi_io.h"
GUARD_MODULE = "lassi.toolchains.ttmetal_guard"
ACCEPTANCE = ("clean_offload", "host_loop_output", "data_movement_only", "compute_beside_host_loop")
SYNTHETIC_LINE = "// SYNTHETIC: written for tests/toolchains/test_ttmetal_guard.py; not tt-metal source"

KERNEL_CREATORS = frozenset({"CreateKernel", "CreateKernelFromString", "KernelDescriptor", "ProgramDescriptor"})
COMPUTE_CONFIGS = frozenset({"ComputeConfig", "ComputeConfigDescriptor"})
DEVICE_READS = frozenset(
    {
        "EnqueueReadMeshBuffer",
        "ReadShard",
        "ReadFromBuffer",
        "ReadFromDeviceL1",
        "ReadFromDeviceDRAMChannel",
        "ReadRegFromDevice",
    }
)
HOST_SUFFIXES = frozenset({".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".inc", ".inl", ".ipp"})
# The spike that records the identifier sets' header lines at the pin, and its rx execs (ttmetal_guard's docstring).
IDENTIFIER_SPIKE = "plans/spikes/p4-guard-identifiers.md"
IDENTIFIER_RX = ("20261004-162204-exec-29e2", "20261004-162216-exec-362c", "20261004-162228-exec-16ba")

PARSE, WARNING, NOTE = "parse", "warning", "note"
TAG = "guard-data-movement-only"
HOST_COMPUTE = "guard-host-compute"
NOT_CHECKED = "guard-not-checked"
SIZE_LIMIT = 4194304
DEPTH_LIMIT = 256

# The messages ttmetal_guard gives (TAG_MESSAGE, NO_KERNEL_MESSAGE, NO_OUTPUT_MESSAGE), as fixed there.
TAG_MESSAGE = (
    "CPU -> TT guard: the host code creates kernels but names no compute kernel config (ComputeConfig or "
    "ComputeConfigDescriptor), so this is a data-movement-only program: tagged, not failed (Harness Contract)"
)
NO_KERNEL_MESSAGE = (
    "CPU -> TT guard: the host code names no kernel-creation call (CreateKernel, CreateKernelFromString, "
    "KernelDescriptor, or ProgramDescriptor) and writes an output, so every output value comes from the host; "
    "host_compute is set (Harness Contract)"
)
NO_OUTPUT_MESSAGE = (
    "CPU -> TT guard: the host code writes no output through lassi_io_write, fwrite, a write on a variable "
    "declared as an output stream, or POSIX write, so host compute was not checked (Harness Contract)"
)
READER_FAILED = "the reader failed with"


def flow_message(source: str, sink: str) -> str:
    """Return rule H's message for an input read at `source` reaching the output written at `sink` (<file>:<line>)."""
    return (
        f"CPU -> TT guard: values read from the inputs at {source} reach the output written at {sink} through host "
        "code, not through a device read; host_compute is set (Harness Contract)"
    )


def unreadable_message(reason: str) -> str:
    """Return the not-checked message of host text that could not be read for `reason`."""
    return (
        f"CPU -> TT guard: the host code could not be read ({reason}), so host compute was not checked "
        "(Harness Contract)"
    )


# ---------------------------------------------------------------------------
# Names this task adds, looked up so a missing one fails its test with a clear message


def import_or_fail(name: str) -> ModuleType:
    """Return the module `name`; fail the test clearly while it does not exist."""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        pytest.fail(f"{name} does not exist ({error}); task P4.12 adds it")


def guard() -> ModuleType:
    """Return lassi.toolchains.ttmetal_guard; fail the test clearly while it is missing."""
    module = import_or_fail(GUARD_MODULE)
    if not callable(getattr(module, "read_host_compute", None)):
        pytest.fail(f"{GUARD_MODULE} has no read_host_compute(files, harness); task P4.12 adds it")
    return module


def reading_class() -> type:
    """Return lassi.core.capabilities.HostComputeReading; fail the test clearly while it is missing."""
    cls = getattr(capabilities_module, "HostComputeReading", None)
    if cls is None:
        pytest.fail("lassi.core.capabilities has no HostComputeReading; task P4.12 adds it")
    return cls


def check_contract(result: Any, *, defect: bool = False) -> None:
    """Assert that `result` is a HostComputeReading within the seam's contract (lassi.core.capabilities)."""
    assert isinstance(result, reading_class()), f"read_host_compute returned {type(result).__name__}"
    assert result.host_compute is True or result.host_compute is False or result.host_compute is None
    assert isinstance(result.diagnostics, tuple), "diagnostics is a tuple"
    for item in result.diagnostics:
        assert isinstance(item, Diagnostic), item
        assert (item.stage, item.severity in (WARNING, NOTE)) == (PARSE, True), item
        assert isinstance(item.code, str) and item.code.startswith("guard-"), item
        assert item.message.isascii(), f"every guard message is plain ASCII: {item.message!r}"
        if not defect:
            assert READER_FAILED not in item.message, f"no test input reaches the reader-failed path: {item.message}"


def read(files: Mapping[str, str], harness: Mapping[str, str] | None = None, *, defect: bool = False) -> Any:
    """Return read_host_compute's reading of `files` with `harness` (none by default), checked for its contract."""
    result = guard().read_host_compute(dict(files), dict(harness or {}))
    check_contract(result, defect=defect)
    return result


def summary(item: Diagnostic) -> tuple[Any, ...]:
    """Return what the contract fixes of a guard Diagnostic: stage, severity, code, file, line, and message."""
    return (item.stage, item.severity, item.code, item.file, item.line, item.message)


def line_of(text: str, needle: str, occurrence: int = 1) -> int:
    """Return the 1-based number of the `occurrence`-th line of `text` that holds `needle`."""
    found = [number for number, line in enumerate(text.splitlines(), start=1) if needle in line]
    assert len(found) >= occurrence, f"{needle!r} occurs on {len(found)} line(s), not {occurrence}"
    return found[occurrence - 1]


def fixture_files(case: str) -> dict[str, str]:
    """Return the files of one fixture case (build-directory path -> text), as a model's reply holds them.

    Each fixture is checked as it is read: main.cpp and at least one kernel
    placeholder, every file plain ASCII and starting with the SYNTHETIC line,
    and the case listed in the folder's README.
    """
    root = FIXTURES / case
    paths = sorted(path for path in root.rglob("*") if path.is_file())
    files = {path.relative_to(root).as_posix(): path.read_bytes().decode("utf-8") for path in paths}
    assert "main.cpp" in files and any(path.startswith("kernels/") for path in files), case
    for path, text in files.items():
        assert text.isascii(), f"{case}/{path} is plain ASCII"
        assert text.splitlines()[0] == SYNTHETIC_LINE, f"{case}/{path} starts with the SYNTHETIC line"
    readme = (FIXTURES / "README.md").read_bytes()
    assert readme.isascii() and f"`{case}/`".encode("ascii") in readme, f"the README lists {case}"
    return files


def harness_header() -> dict[str, str]:
    """Return the harness file a build gets: assets/harness/c/lassi_io.h as lassi_io.h."""
    return {"lassi_io.h": HARNESS_HEADER.read_bytes().decode("utf-8")}


@pytest.fixture(params=["no-harness", "harness-header"])
def harness(request: pytest.FixtureRequest) -> dict[str, str]:
    """Return no harness files, or the real lassi_io.h, which the guard then reads through the include."""
    return {} if request.param == "no-harness" else harness_header()


# ---------------------------------------------------------------------------
# SYNTHETIC rule-case programs (none is compiled or run)

PRELUDE = (
    SYNTHETIC_LINE
    + "\n"
    + """\
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <execution>
#include <fstream>
#include <numeric>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>

#include "lassi_io.h"

using namespace tt::tt_metal;
"""
)
MAIN_OPEN = """\
int main(int argc, char** argv) {
    if (argc != 4) {
        return 2;
    }
    lassi_io_array a = {};
    lassi_io_array b = {};
    if (lassi_io_read(argv[1], &a) != 0) {
        return 1;
    }
    if (lassi_io_read(argv[2], &b) != 0) {
        return 1;
    }
    const uint32_t n = static_cast<uint32_t>(a.count);
    const float* af = static_cast<const float*>(a.data);
    const float* bf = static_cast<const float*>(b.data);
    std::vector<float> c(n);
    std::vector<float> res(n);
    auto device = distributed::MeshDevice::create_unit_mesh(0);
    distributed::MeshCommandQueue& cq = device->mesh_command_queue();
    distributed::DeviceLocalBufferConfig dram{.page_size = 4 * n, .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig whole{.size = 4 * n};
    auto in_buf = distributed::MeshBuffer::create(whole, dram, device.get());
    auto c_buf = distributed::MeshBuffer::create(whole, dram, device.get());
"""
# Kernels with a compute config, so neither rule K nor the tag applies and rule H decides.
KERNELS = """\
    Program program = CreateProgram();
    const CoreCoord core = {0, 0};
    KernelHandle reader = CreateKernel(program, "kernels/reader.cpp", core, DataMovementConfig{});
    KernelHandle compute = CreateKernel(program, "kernels/compute.cpp", core, ComputeConfig{});
    SetRuntimeArgs(program, reader, core, {n});
    SetRuntimeArgs(program, compute, core, {n});
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(device->shape()), std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
"""
# One data-movement kernel and no compute config: the tag applies.
KERNELS_DM = """\
    Program program = CreateProgram();
    const CoreCoord core = {0, 0};
    KernelHandle mover = CreateKernel(program, "kernels/mover.cpp", core, DataMovementConfig{});
    SetRuntimeArgs(program, mover, core, {n});
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(device->shape()), std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
"""
READ_BACK = "    distributed::EnqueueReadMeshBuffer(cq, res, c_buf, true);\n"
READ_INTO_C = "    distributed::EnqueueReadMeshBuffer(cq, c, c_buf, true);\n"
DEVICE_TO_C = """\
    for (uint32_t i = 0; i < n; ++i) {
        c[i] = res[i];
    }
"""
HOST_TO_C = """\
    for (uint32_t i = 0; i < n; ++i) {
        c[i] = af[i] + bf[i];
    }
"""
CLEAN_BODY = READ_BACK + DEVICE_TO_C
WRITE_C = """\
    uint64_t dims[1] = {n};
    if (lassi_io_write(argv[3], "c", LASSI_IO_F32, 1u, dims, c.data()) != 0) {
        return 1;
    }
"""
END = """\
    return 0;
}
"""
READ_A = "lassi_io_read(argv[1]"
READ_B = "lassi_io_read(argv[2]"
WRITE_NEEDLE = "lassi_io_write(argv[3]"

# A helper that writes its third argument from the first two, and one with the same signature that does not.
VIOLATING_OPS = (
    SYNTHETIC_LINE
    + "\n"
    + """\
#pragma once
#include <cstdint>
static void combine(const float* x, const float* y, float* z, uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
        z[i] = x[i] + y[i];
    }
}
"""
)
CLEAN_OPS = (
    SYNTHETIC_LINE
    + "\n"
    + """\
#pragma once
#include <cstdint>
static void combine(const float* x, const float* y, float* z, uint32_t count) {
    (void)x;
    (void)y;
    for (uint32_t i = 0; i < count; ++i) {
        z[i] = 0.0f;
    }
}
"""
)
COMBINE_BODY = "    combine(af, bf, c.data(), n);\n"


def program(
    body: str = "", *, kernels: str = KERNELS, helpers: str = "", includes: str = "", tail: str = WRITE_C
) -> str:
    """Return a SYNTHETIC host program: the prelude, `includes`, `helpers`, then main with `kernels`, `body`, `tail`.

    main reads the inputs a and b with lassi_io_read (af and bf alias their
    data), declares the output vector c and the device-result vector res,
    and creates a device, its queue, and two buffers; by default it creates
    kernels with a compute config, and `tail` writes c with lassi_io_write.
    """
    return PRELUDE + includes + helpers + MAIN_OPEN + kernels + body + tail + END


def assert_clear(result: Any, *, tagged_at: int | None = None) -> None:
    """Assert a False reading with no diagnostic, or with only the tag at line `tagged_at` of main.cpp."""
    assert result.host_compute is False, result
    if tagged_at is None:
        assert result.diagnostics == (), result.diagnostics
    else:
        assert [summary(item) for item in result.diagnostics] == [tag_summary("main.cpp", tagged_at)]


def tag_summary(file: str, line: int) -> tuple[Any, ...]:
    """Return the expected summary of the tag at `file`:`line`."""
    return (PARSE, WARNING, TAG, file, line, TAG_MESSAGE)


def flow_summary(
    text: str, *, source: str = READ_A, sink: str = WRITE_NEEDLE, file: str = "main.cpp"
) -> tuple[Any, ...]:
    """Return the expected summary of rule H's note for the program `text` in `file`."""
    source_line, sink_line = line_of(text, source), line_of(text, sink)
    message = flow_message(f"{file}:{source_line}", f"{file}:{sink_line}")
    return (PARSE, NOTE, HOST_COMPUTE, file, sink_line, message)


def assert_flow(
    result: Any, text: str, *, source: str = READ_A, sink: str = WRITE_NEEDLE, file: str = "main.cpp"
) -> None:
    """Assert a True reading from rule H with exactly its one note, naming `source` and `sink` lines of `text`."""
    assert result.host_compute is True, result
    assert [summary(item) for item in result.diagnostics] == [flow_summary(text, source=source, sink=sink, file=file)]


def assert_no_kernel(result: Any, text: str, *, sink: str = WRITE_NEEDLE, file: str = "main.cpp") -> None:
    """Assert a True reading from rule K: exactly one note at the first sink, and no tag."""
    assert result.host_compute is True, result
    expected = (PARSE, NOTE, HOST_COMPUTE, file, line_of(text, sink), NO_KERNEL_MESSAGE)
    assert [summary(item) for item in result.diagnostics] == [expected]


def no_output_summary() -> tuple[Any, ...]:
    """Return the expected summary of the not-checked note of host text that writes no output."""
    return (PARSE, NOTE, NOT_CHECKED, None, None, NO_OUTPUT_MESSAGE)


def unreadable_summary(reason: str) -> tuple[Any, ...]:
    """Return the expected summary of the not-checked note of host text that could not be read for `reason`."""
    return (PARSE, NOTE, NOT_CHECKED, None, None, unreadable_message(reason))


# ---------------------------------------------------------------------------
# The module, its identifier sets, and the fixtures


def test_the_identifier_sets_are_the_pinned_api_names() -> None:
    module = guard()
    assert frozenset(module.KERNEL_CREATORS) == KERNEL_CREATORS
    assert frozenset(module.COMPUTE_CONFIGS) == COMPUTE_CONFIGS
    assert frozenset(module.DEVICE_READS) == DEVICE_READS
    assert frozenset(module.HOST_SUFFIXES) == HOST_SUFFIXES


def test_the_module_records_the_rules_and_where_each_identifier_set_comes_from() -> None:
    module = guard()
    source = inspect.getsource(module)
    spike = (REPO / IDENTIFIER_SPIKE).read_bytes().decode("ascii")
    assert IDENTIFIER_SPIKE in source, f"the identifier sets cite {IDENTIFIER_SPIKE}"
    for rx in IDENTIFIER_RX:
        assert rx in source, f"the identifier sets cite rx {rx}"
        assert rx in spike, f"{IDENTIFIER_SPIKE} records rx {rx}"
    doc = module.__doc__ or ""
    for name in sorted(KERNEL_CREATORS | COMPUTE_CONFIGS | DEVICE_READS):
        assert name in doc, f"the module docstring states the full rules, {name} included"


@pytest.mark.parametrize("name", ["lassi.toolchains._cxx_scan", "lassi.toolchains._cxx_flow"])
def test_the_generic_readers_name_nothing_from_tt_metal(name: str) -> None:
    source = inspect.getsource(import_or_fail(name))
    named = sorted(word for word in KERNEL_CREATORS | COMPUTE_CONFIGS | DEVICE_READS if word in source)
    assert named == [], f"{name} is generic C and C++ analysis; the tt-metal names live in {GUARD_MODULE}"


# ---------------------------------------------------------------------------
# The four acceptance programs (plans/p4-ttsim.md, P4.12)


def test_clean_offload_reads_false_with_no_diagnostic(harness: dict[str, str]) -> None:
    assert_clear(read(fixture_files("clean_offload"), harness))


def test_host_loop_writing_outputs_reads_true(harness: dict[str, str]) -> None:
    files = fixture_files("host_loop_output")
    assert_flow(read(files, harness), files["main.cpp"], sink="lassi_io_write(")


def test_data_movement_only_is_tagged_not_failed(harness: dict[str, str]) -> None:
    files = fixture_files("data_movement_only")
    assert_clear(read(files, harness), tagged_at=line_of(files["main.cpp"], "CreateKernel("))


def test_compute_kernel_beside_host_loop_that_writes_no_output_reads_false(harness: dict[str, str]) -> None:
    assert_clear(read(fixture_files("compute_beside_host_loop"), harness))


def test_the_harness_header_alone_writes_no_output() -> None:
    # lassi_io.h holds fwrite and fread inside its own lassi_io_* definitions, which the reader skips (_cxx_flow),
    # so a program that includes it and writes nothing writes no recognized output.
    text = program(CLEAN_BODY, tail="")
    result = read({"main.cpp": text}, harness_header())
    assert result.host_compute is None
    assert [summary(item) for item in result.diagnostics] == [no_output_summary()]


# ---------------------------------------------------------------------------
# Rules K and T (ttmetal_guard's module docstring, The rules)


def test_no_kernel_creator_with_an_output_write_is_host_compute() -> None:
    body = "    std::transform(af, af + n, bf, c.begin(), [](float x, float y) { return x + y; });\n"
    text = program(body, kernels="")
    assert_no_kernel(read({"main.cpp": text}), text)


def test_no_kernel_creator_with_a_device_result_written_is_still_host_compute() -> None:
    # Rule K reads names only: with no kernel created, no device work made any output value.
    text = program(CLEAN_BODY, kernels="")
    assert_no_kernel(read({"main.cpp": text}), text)


def test_no_kernel_creator_and_no_output_write_is_not_checked() -> None:
    result = read({"main.cpp": program(HOST_TO_C, kernels="", tail="")})
    assert result.host_compute is None
    assert [summary(item) for item in result.diagnostics] == [no_output_summary()]


@pytest.mark.parametrize(
    "hidden",
    [
        '    // CreateKernel(program, "kernels/compute.cpp", core, ComputeConfig{});\n',
        '    /* CreateKernel(program, "kernels/compute.cpp", core, ComputeConfig{}); */\n',
        '    const char* note = "CreateKernel(program, core, ComputeConfig{})";\n',
        '    const char* raw = R"x(CreateKernel(program, core, ComputeConfig{}))x";\n',
    ],
    ids=["line-comment", "block-comment", "string", "raw-string"],
)
def test_a_kernel_creator_in_a_comment_or_a_literal_does_not_count(hidden: str) -> None:
    text = program(hidden + CLEAN_BODY, kernels="")
    assert_no_kernel(read({"main.cpp": text}), text)


def test_kernel_names_in_define_bodies_count() -> None:
    helpers = "#define MAKE_COMPUTE(p, f, c) CreateKernel(p, f, c, ComputeConfig{})\n"
    assert_clear(read({"main.cpp": program(CLEAN_BODY, kernels="", helpers=helpers)}))


@pytest.mark.parametrize(
    "include",
    ["#include <my/CreateKernel.hpp>\n", '#include "my/CreateKernel.hpp"\n', "#  include <tt/KernelDescriptor.h>\n"],
    ids=["angle", "quoted", "angle-spaced"],
)
def test_a_name_in_an_include_line_does_not_count(include: str) -> None:
    # A header name is no identifier: the path segments of an `#include` line name no kernel creator or config.
    text = program(CLEAN_BODY, kernels="", includes=include)
    assert_no_kernel(read({"main.cpp": text}), text)


def test_a_kernel_creator_inside_if_0_does_not_count() -> None:
    text = program(CLEAN_BODY, kernels="#if 0\n" + KERNELS + "#endif\n")
    assert_no_kernel(read({"main.cpp": text}), text)


def test_a_kernel_creator_in_the_else_branch_of_if_0_counts() -> None:
    text = program(CLEAN_BODY, kernels="#if 0\n    int unused = 0;\n#else\n" + KERNELS + "#endif\n")
    assert_clear(read({"main.cpp": text}))


@pytest.mark.parametrize(
    "line",
    [
        '    KernelHandle made = CreateKernel(program, "kernels/mover.cpp", core, DataMovementConfig{});\n',
        '    KernelHandle made = CreateKernelFromString(program, "void kernel_main() {}", core, {});\n',
        "    KernelDescriptor made{};\n",
        "    ProgramDescriptor made{};\n",
    ],
    ids=sorted(KERNEL_CREATORS),
)
def test_each_kernel_creator_alone_counts_as_kernel_creation(line: str) -> None:
    kernels = "    Program program = CreateProgram();\n    const CoreCoord core = {0, 0};\n" + line
    text = program(CLEAN_BODY, kernels=kernels)
    assert_clear(read({"main.cpp": text}), tagged_at=line_of(text, line.strip()))


def test_compute_config_descriptor_counts_as_a_compute_config() -> None:
    kernels = "    ProgramDescriptor described{};\n    ComputeConfigDescriptor config{};\n"
    assert_clear(read({"main.cpp": program(CLEAN_BODY, kernels=kernels)}))


def test_rule_k_reports_the_first_sink() -> None:
    tail = WRITE_C + '    lassi_io_write(argv[3], "d", LASSI_IO_F32, 1u, dims, res.data());\n'
    text = program(HOST_TO_C, kernels="", tail=tail)
    assert_no_kernel(read({"main.cpp": text}), text)


def test_rule_k_reports_the_first_sink_in_sorted_file_order() -> None:
    first = SYNTHETIC_LINE + "\n#include \"lassi_io.h\"\n" + (
        "void save_a(const char* path, const uint64_t* dims, const float* data) {\n"
        '    lassi_io_write(path, "a", LASSI_IO_F32, 1u, dims, data);\n'
        "}\n"
    )
    files = {"b_main.cpp": program(HOST_TO_C, kernels="", tail=""), "a_save.cpp": first}
    result = read(files)
    expected = (PARSE, NOTE, HOST_COMPUTE, "a_save.cpp", line_of(first, "lassi_io_write("), NO_KERNEL_MESSAGE)
    assert result.host_compute is True and [summary(item) for item in result.diagnostics] == [expected]


def test_the_tag_sits_beside_a_host_compute_flow() -> None:
    text = program(HOST_TO_C, kernels=KERNELS_DM)
    result = read({"main.cpp": text})
    assert result.host_compute is True
    expected = [tag_summary("main.cpp", line_of(text, "CreateKernel(")), flow_summary(text)]
    assert [summary(item) for item in result.diagnostics] == expected, "the tag first, then the host-compute note"


def test_the_tag_sits_beside_a_reading_that_is_not_checked() -> None:
    text = program(CLEAN_BODY, kernels=KERNELS_DM, tail="")
    result = read({"main.cpp": text})
    assert result.host_compute is None
    expected = [tag_summary("main.cpp", line_of(text, "CreateKernel(")), no_output_summary()]
    assert [summary(item) for item in result.diagnostics] == expected


def test_the_tag_sits_at_the_first_creator_in_a_model_file() -> None:
    early = SYNTHETIC_LINE + "\ninline void early_setup() {\n    KernelDescriptor described{};\n}\n"
    text = program(CLEAN_BODY, kernels=KERNELS_DM, includes='#include "aaa/early.h"\n')
    result = read({"main.cpp": text}, {"aaa/early.h": early})
    assert_clear(result, tagged_at=line_of(text, "CreateKernel("))


# ---------------------------------------------------------------------------
# Rule H through library calls (_cxx_flow's module docstring: Standard calls and Member calls)

ACCUMULATE_TAIL = """\
    if (lassi_io_write(argv[3], "total", LASSI_IO_F32, 0u, nullptr, &total) != 0) {
        return 1;
    }
"""
WRITE_OUT = """\
    uint64_t dims[1] = {n};
    if (lassi_io_write(argv[3], "c", LASSI_IO_F32, 1u, dims, out.data()) != 0) {
        return 1;
    }
"""
# (id, body, tail, the input whose read the note names, or None for a False reading)
LIBRARY_CASES: list[tuple[str, str, str, str | None]] = [
    ("memcpy", "    std::memcpy(c.data(), af, n * sizeof(float));\n", WRITE_C, READ_A),
    ("memmove", "    memmove(c.data(), af, n * sizeof(float));\n", WRITE_C, READ_A),
    ("copy", "    std::copy(af, af + n, c.begin());\n", WRITE_C, READ_A),
    ("copy_n", "    std::copy_n(bf, n, c.begin());\n", WRITE_C, READ_B),
    (
        "transform-binary-lambda",
        "    std::transform(af, af + n, bf, c.begin(), [](float x, float y) { return x + y; });\n",
        WRITE_C,
        READ_A,
    ),
    ("transform-unary", "    std::transform(af, af + n, c.begin(), [](float x) { return 2 * x; });\n", WRITE_C, READ_A),
    (
        "transform-execution-policy",
        "    std::transform(std::execution::par, af, af + n, bf, c.begin(), std::plus<float>{});\n",
        WRITE_C,
        READ_A,
    ),
    (
        "ranges-copy",
        "    std::vector<float> in_a(af, af + n);\n    std::ranges::copy(in_a, c.begin());\n",
        WRITE_C,
        READ_A,
    ),
    (
        "ranges-transform",
        "    std::ranges::transform(std::vector<float>(af, af + n), c.begin(), negate);\n",
        WRITE_C,
        READ_A,
    ),
    ("accumulate-scalar", "    float total = std::accumulate(af, af + n, 0.0f);\n", ACCUMULATE_TAIL, READ_A),
    (
        "for-each-reference",
        "    uint32_t k = 0;\n    std::for_each(c.begin(), c.end(), [&](float& v) { v = af[k] + bf[k]; ++k; });\n",
        WRITE_C,
        READ_A,
    ),
    ("generate-capturing", "    std::generate(c.begin(), c.end(), [&]() { return af[0]; });\n", WRITE_C, READ_A),
    (
        "push-back",
        "    std::vector<float> out;\n    for (uint32_t i = 0; i < n; ++i) {\n        out.push_back(af[i]);\n    }\n",
        WRITE_OUT,
        READ_A,
    ),
    ("assign", "    c.assign(af, af + n);\n", WRITE_C, READ_A),
    ("insert", "    c.insert(c.begin(), af, af + n);\n", WRITE_C, READ_A),
    ("std-swap", "    std::vector<float> in_a(af, af + n);\n    std::swap(in_a, c);\n", WRITE_C, READ_A),
    ("member-swap", "    std::vector<float> in_a(af, af + n);\n    c.swap(in_a);\n", WRITE_C, READ_A),
    ("fill-constant", "    std::fill(c.begin(), c.end(), 1.0f);\n", WRITE_C, None),
    ("iota-constant", "    std::iota(c.begin(), c.end(), 0.0f);\n", WRITE_C, None),
    ("copy-device-result", READ_BACK + "    std::copy(res.begin(), res.end(), c.begin());\n", WRITE_C, None),
    (
        "transform-device-result",
        READ_BACK + "    std::transform(res.begin(), res.end(), c.begin(), [](float v) { return v * 0.5f; });\n",
        WRITE_C,
        None,
    ),
    ("memcpy-device-result", READ_BACK + "    std::memcpy(c.data(), res.data(), n * sizeof(float));\n", WRITE_C, None),
]


@pytest.mark.parametrize(
    ("body", "tail", "source"), [case[1:] for case in LIBRARY_CASES], ids=[case[0] for case in LIBRARY_CASES]
)
def test_flow_through_library_calls(body: str, tail: str, source: str | None) -> None:
    text = program(body, tail=tail)
    result = read({"main.cpp": text})
    if source is None:
        assert_clear(result)
    else:
        assert_flow(result, text, source=source)


def test_an_own_function_named_like_a_standard_call_wins_over_the_standard_rule() -> None:
    helpers = (
        "static void copy(const float* from, float* to, uint32_t count) {\n"
        "    (void)from;\n"
        "    for (uint32_t i = 0; i < count; ++i) {\n"
        "        to[i] = 0.0f;\n"
        "    }\n"
        "}\n"
    )
    assert_clear(read({"main.cpp": program("    copy(af, c.data(), n);\n", helpers=helpers)}))
    # A std::-qualified callee never matches an own definition, so the standard rule applies to it.
    text = program("    std::copy(af, af + n, c.begin());\n", helpers=helpers)
    assert_flow(read({"main.cpp": text}), text)


# ---------------------------------------------------------------------------
# Own functions: summaries (_cxx_flow's module docstring, Own functions)

ADD_HELPER = """\
static void add(const float* x, const float* y, float* z, uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
        z[i] = x[i] + y[i];
    }
}
"""


def test_a_helper_writing_its_output_argument_from_the_inputs_is_host_compute() -> None:
    text = program("    add(af, bf, c.data(), n);\n", helpers=ADD_HELPER)
    assert_flow(read({"main.cpp": text}), text)


SPAN_INCLUDE = "#include <span>\n"
SPAN_ADD = """\
static void add(std::span<const float> x, std::span<const float> y, std::span<float> z) {
    for (std::size_t i = 0; i < z.size(); ++i) {
        z[i] = x[i] + y[i];
    }
}
"""
SPAN_READ = """\
static void read_into(const std::shared_ptr<distributed::MeshBuffer>& from, std::span<float> to) {
    ReadFromBuffer(*from, reinterpret_cast<uint8_t*>(to.data()));
}
"""


def test_a_view_parameter_carries_a_write_back_to_its_argument() -> None:
    # A parameter whose type's head is a view (std::span) points into its argument's storage, as a pointer does, in
    # an own function and in a named lambda (the P4.12 commit audit found both missed).
    text = program("    add({af, n}, {bf, n}, c);\n", helpers=SPAN_ADD, includes=SPAN_INCLUDE)
    assert_flow(read({"main.cpp": text}), text)
    clean = program(READ_BACK + "    add(res, res, c);\n", helpers=SPAN_ADD, includes=SPAN_INCLUDE)
    assert_clear(read({"main.cpp": clean}))
    named = (
        "    auto copy_into = [](std::span<const float> x, std::span<float> z) {\n"
        "        for (std::size_t i = 0; i < z.size(); ++i) {\n"
        "            z[i] = x[i];\n"
        "        }\n"
        "    };\n"
        "    copy_into(std::span<const float>(af, n), c);\n"
    )
    text = program(named, includes=SPAN_INCLUDE)
    assert_flow(read({"main.cpp": text}), text)


def test_a_device_read_through_a_view_parameter_clears_its_argument() -> None:
    staged = "    std::vector<float> staged(af, af + n);\n    read_into(c_buf, staged);\n    c = staged;\n"
    assert_clear(read({"main.cpp": program(staged, helpers=SPAN_READ, includes=SPAN_INCLUDE)}))
    text = program(staged.replace("read_into(c_buf, staged);", "read_into(c_buf, res);"), helpers=SPAN_READ,
                   includes=SPAN_INCLUDE)
    assert_flow(read({"main.cpp": text}), text)


def test_a_shared_reorder_used_for_staging_and_for_the_device_result_is_clear() -> None:
    helpers = (
        "static void reorder(const std::vector<float>& src, std::vector<float>& dst) {\n"
        "    for (std::size_t i = 0; i < src.size(); ++i) {\n"
        "        dst[i] = src[(i * 7) % src.size()];\n"
        "    }\n"
        "}\n"
    )
    body = (
        "    std::vector<float> a_in(af, af + n);\n"
        "    std::vector<float> a_tiled(n);\n"
        "    reorder(a_in, a_tiled);\n"
        "    distributed::EnqueueWriteMeshBuffer(cq, in_buf, a_tiled, false);\n"
        + READ_BACK
        + "    reorder(res, c);\n"
    )
    assert_clear(read({"main.cpp": program(body, helpers=helpers)}))


def test_a_helper_returning_input_derived_values_by_value_is_host_compute() -> None:
    helpers = (
        "static std::vector<float> sum(const float* x, const float* y, uint32_t count) {\n"
        "    std::vector<float> out(count);\n"
        "    for (uint32_t i = 0; i < count; ++i) {\n"
        "        out[i] = x[i] + y[i];\n"
        "    }\n"
        "    return out;\n"
        "}\n"
    )
    text = program("    c = sum(af, bf, n);\n", helpers=helpers)
    assert_flow(read({"main.cpp": text}), text)


@pytest.mark.parametrize("where", ["in-class", "out-of-class"])
def test_a_member_method_writing_outputs_from_inputs_is_host_compute(where: str) -> None:
    method_body = (
        "{\n"
        "    for (uint32_t i = 0; i < count; ++i) {\n"
        "        z[i] = scale * (x[i] + y[i]);\n"
        "    }\n"
        "}\n"
    )
    signature = "void {name}(const float* x, const float* y, float* z, uint32_t count) const"
    head = "struct Runner {\n    float scale = 1.0f;\n    " + signature.format(name="add")
    if where == "in-class":
        helpers = head + " " + method_body + "};\n"
    else:
        helpers = head + ";\n};\n" + signature.format(name="Runner::add") + " " + method_body
    text = program("    Runner r;\n    r.add(af, bf, c.data(), n);\n", helpers=helpers)
    assert_flow(read({"main.cpp": text}), text)


def test_a_method_returning_a_data_member_carries_what_another_method_stored() -> None:
    helpers = (
        "struct Holder {\n"
        "    std::vector<float> values;\n"
        "    void load(const float* x, uint32_t count) {\n"
        "        values.assign(x, x + count);\n"
        "    }\n"
        "    std::vector<float> get() const {\n"
        "        return this->values;\n"
        "    }\n"
        "};\n"
    )
    text = program("    Holder h;\n    h.load(af, n);\n    c = h.get();\n", helpers=helpers)
    assert_flow(read({"main.cpp": text}), text)


def test_an_unused_cpu_reference_function_is_not_host_compute_of_the_outputs() -> None:
    helpers = (
        "static int cpu_reference(const char* in_a, const char* in_b, const char* out) {\n"
        "    lassi_io_array x = {};\n"
        "    lassi_io_array y = {};\n"
        "    if (lassi_io_read(in_a, &x) != 0 || lassi_io_read(in_b, &y) != 0) {\n"
        "        return 1;\n"
        "    }\n"
        "    const float* xf = static_cast<const float*>(x.data);\n"
        "    const float* yf = static_cast<const float*>(y.data);\n"
        "    std::vector<float> z(x.count);\n"
        "    for (uint64_t i = 0; i < x.count; ++i) {\n"
        "        z[i] = xf[i] + yf[i];\n"
        "    }\n"
        "    uint64_t shape[1] = {x.count};\n"
        '    return lassi_io_write(out, "c", LASSI_IO_F32, 1u, shape, z.data());\n'
        "}\n"
    )
    assert_clear(read({"main.cpp": program(CLEAN_BODY, helpers=helpers)}))


@pytest.mark.parametrize("header", ["kernels/helpers.h", "kernels/helpers.cpp"])
def test_host_code_moved_into_an_included_kernels_file_is_read(header: str) -> None:
    text = program(COMBINE_BODY, includes=f'#include "{header}"\n')
    assert_flow(read({"main.cpp": text, header: VIOLATING_OPS}), text)


def test_a_kernels_file_that_no_host_file_includes_is_never_read() -> None:
    kernel = (
        SYNTHETIC_LINE
        + "\n#include \"lassi_io.h\"\n"
        + "void kernel_main() {\n"
        + "    lassi_io_array x = {};\n"
        + '    lassi_io_read("in.bin", &x);\n'
        + "    uint64_t shape[1] = {x.count};\n"
        + '    lassi_io_write("out.bin", "c", LASSI_IO_F32, 1u, shape, x.data);\n'
        + "    ComputeConfig config{};\n"
        + "    CreateKernel(config);\n"
        + "}\n"
    )
    assert_clear(read({"main.cpp": program(CLEAN_BODY), "kernels/compute.cpp": kernel}))
    quiet = read({"main.cpp": program(HOST_TO_C, kernels="", tail=""), "kernels/compute.cpp": kernel})
    assert quiet.host_compute is None, "the kernel file's write and kernel names are never read"
    assert [summary(item) for item in quiet.diagnostics] == [no_output_summary()]


def test_a_by_value_parameter_never_carries_values_back() -> None:
    helpers = (
        "static void fill_copy(const float* x, std::vector<float> c, uint32_t count) {\n"
        "    for (uint32_t i = 0; i < count; ++i) {\n"
        "        c[i] = x[i];\n"
        "    }\n"
        "}\n"
    )
    assert_clear(read({"main.cpp": program("    fill_copy(af, c, n);\n" + CLEAN_BODY, helpers=helpers)}))


def test_a_file_scope_variable_carries_values_between_helpers() -> None:
    helpers = (
        "static std::vector<float> g_scratch;\n"
        "static void keep(const float* x, uint32_t count) {\n"
        "    g_scratch.assign(x, x + count);\n"
        "}\n"
        "static void emit(float* out, uint32_t count) {\n"
        "    for (uint32_t i = 0; i < count; ++i) {\n"
        "        out[i] = g_scratch[i];\n"
        "    }\n"
        "}\n"
    )
    text = program("    keep(af, n);\n    emit(c.data(), n);\n", helpers=helpers)
    assert_flow(read({"main.cpp": text}), text)


def test_a_helper_defined_in_another_host_source_is_summarized() -> None:
    util = SYNTHETIC_LINE + '\n#include "util.h"\n' + VIOLATING_OPS.split("#include <cstdint>\n", 1)[1]
    declaration = "void combine(const float* x, const float* y, float* z, uint32_t count);\n"
    header = SYNTHETIC_LINE + "\n#pragma once\n#include <cstdint>\n" + declaration
    text = program(COMBINE_BODY, includes='#include "util.h"\n')
    files = {"main.cpp": text, "util.cpp": util.replace("static void", "void"), "util.h": header}
    assert_flow(read(files), text)


def test_recursive_helpers_end_and_keep_their_flows() -> None:
    helpers = (
        "static void ping(const float* x, float* y, uint32_t k);\n"
        "static void pong(const float* x, float* y, uint32_t k) {\n"
        "    if (k == 0) {\n"
        "        return;\n"
        "    }\n"
        "    y[k - 1] = x[k - 1];\n"
        "    ping(x, y, k - 1);\n"
        "}\n"
        "static void ping(const float* x, float* y, uint32_t k) {\n"
        "    if (k == 0) {\n"
        "        return;\n"
        "    }\n"
        "    pong(x, y, k);\n"
        "}\n"
    )
    text = program("    ping(af, c.data(), n);\n", helpers=helpers)
    assert_flow(read({"main.cpp": text}), text)


def test_without_main_every_function_is_checked() -> None:
    runner = (
        SYNTHETIC_LINE
        + "\n#include \"lassi_io.h\"\n#include <vector>\n"
        + "int run(const char* in_path, const char* out_path) {\n"
        + "    lassi_io_array x = {};\n"
        + "    if (lassi_io_read(in_path, &x) != 0) {\n"
        + "        return 1;\n"
        + "    }\n"
        + "    const float* xf = static_cast<const float*>(x.data);\n"
        + "    KernelHandle unused = CreateKernel(ComputeConfig{});\n"
        + "    std::vector<float> y(xf, xf + x.count);\n"
        + "    uint64_t shape[1] = {x.count};\n"
        + '    return lassi_io_write(out_path, "y", LASSI_IO_F32, 1u, shape, y.data());\n'
        + "}\n"
    )
    result = read({"runner.cpp": runner})
    assert_flow(result, runner, source="lassi_io_read(", sink="lassi_io_write(", file="runner.cpp")


HOLDER = """\
struct Holder {
    std::vector<float> values;
    void load(const float* x, uint32_t count) {
        values.assign(x, x + count);
    }
    std::vector<float> get() const {
        return this->values;
    }
};
"""
TILE = """\
struct Tile {
    float bias;
    float zero() const {
        return 0.0f;
    }
};
"""


def test_an_own_method_reads_its_receiver_only_through_its_return() -> None:
    # The receiver of an own method call reaches the value only when the method returns what THIS holds.
    body = (
        "    std::vector<Tile> tiles(n);\n"
        "    for (uint32_t i = 0; i < n; ++i) {\n"
        "        tiles[i].bias = af[i];\n"
        "    }\n"
        + READ_INTO_C
        + "    for (uint32_t i = 0; i < n; ++i) {\n"
        "        c[i] += tiles[i].zero();\n"
        "    }\n"
    )
    assert_clear(read({"main.cpp": program(body, helpers=TILE)}))
    stored = "    std::vector<Holder> holders(1);\n    holders[0].load(af, n);\n    c = holders[0].get();\n"
    text = program(stored, helpers=HOLDER)
    assert_flow(read({"main.cpp": text}), text)


@pytest.mark.parametrize(
    ("helpers", "body", "tail"),
    [
        (
            "struct Buf {\n    std::vector<float> v;\n    float* data() {\n        return v.data();\n    }\n};\n",
            HOST_TO_C,
            WRITE_C,
        ),
        (
            "struct HostBuffer {\n    std::vector<float> values;\n    float* data() {\n"
            "        return values.data();\n    }\n};\n",
            "    HostBuffer out;\n    out.values.resize(n);\n    for (uint32_t i = 0; i < n; ++i) {\n"
            "        out.values[i] = af[i] * 2.0f;\n    }\n",
            WRITE_OUT,
        ),
    ],
    ids=["unrelated-class", "wrapper-class"],
)
def test_an_own_method_is_matched_only_on_a_receiver_that_can_be_a_host_class(
    helpers: str, body: str, tail: str
) -> None:
    # A std::vector receiver keeps the library reading of c.data(), so the program's own data() hides nothing.
    text = program(body, helpers=helpers, tail=tail)
    assert_flow(read({"main.cpp": text}), text)


READ_BACK_INTO = """\
static std::vector<float> read_back_into(distributed::MeshCommandQueue& queue,
                                         std::shared_ptr<distributed::MeshBuffer>& buffer,
                                         std::vector<float> host) {
    distributed::EnqueueReadMeshBuffer(queue, host, buffer, true);
    return host;
}
"""


def test_a_strong_update_from_an_own_call_reads_only_its_return_sources() -> None:
    # buf is passed to the helper, but the helper's return carries the device read, not buf: buf is killed.
    body = "    std::vector<float> buf(af, af + n);\n    buf = read_back_into(cq, c_buf, buf);\n    c = buf;\n"
    assert_clear(read({"main.cpp": program(body, helpers=READ_BACK_INTO)}))
    identity = "static std::vector<float> identity(std::vector<float> v) {\n    return v;\n}\n"
    body = "    std::vector<float> buf(af, af + n);\n    buf = identity(buf);\n    c = buf;\n"
    text = program(body, helpers=identity)
    assert_flow(read({"main.cpp": text}), text)


SCALED = """\
struct Scaled {
    float v;
    Scaled(float x) : v(x * 2.0f) {}
};
"""


def test_a_constructed_temporary_carries_its_arguments() -> None:
    text = program("    for (uint32_t i = 0; i < n; ++i) {\n        c[i] = Scaled(af[i]).v;\n    }\n", helpers=SCALED)
    assert_flow(read({"main.cpp": text}), text)
    clean = READ_BACK + "    for (uint32_t i = 0; i < n; ++i) {\n        c[i] = Scaled(res[i]).v;\n    }\n"
    assert_clear(read({"main.cpp": program(clean, helpers=SCALED)}))


@pytest.mark.parametrize(
    "arguments", ["af, argv[3], n", "static_cast<const float*>(a.data), argv[3], static_cast<uint32_t>(a.count)"]
)
def test_a_constructor_run_by_a_declaration_is_applied(arguments: str) -> None:
    helpers = (
        "struct Job {\n"
        "    Job(const float* x, const char* out, uint32_t count) {\n"
        "        std::vector<float> y(count);\n"
        "        for (uint32_t i = 0; i < count; ++i) {\n"
        "            y[i] = x[i] * 2.0f;\n"
        "        }\n"
        "        uint64_t shape[1] = {count};\n"
        '        lassi_io_write(out, "c", LASSI_IO_F32, 1u, shape, y.data());\n'
        "    }\n"
        "};\n"
    )
    text = program(f"    Job job({arguments});\n    (void)job;\n", helpers=helpers, tail="")
    assert_flow(read({"main.cpp": text}), text, sink="Job job(")


@pytest.mark.parametrize(
    "body",
    [
        "    float x = 0.0f;\n    float y = 0.0f;\n    x = y = af[0];\n    c[0] = x;\n",
        "    c = [&]() {\n        std::vector<float> r(n);\n        for (uint32_t i = 0; i < n; ++i) {\n"
        "            r[i] = af[i] * 2.0f;\n        }\n        return r;\n    }();\n",
    ],
    ids=["assignment-chain", "immediately-invoked-lambda"],
)
def test_an_inner_assignment_gives_its_right_side_to_the_outer_value(body: str) -> None:
    text = program(body)
    assert_flow(read({"main.cpp": text}), text)


# ---------------------------------------------------------------------------
# Kills, aliases, and scoping (_cxx_scan's Structure; _cxx_flow's Value edges and Visibility)


def test_an_alias_written_from_inputs_reaches_the_output() -> None:
    body = "    float* o = c.data();\n    for (uint32_t i = 0; i < n; ++i) {\n        o[i] = af[i] + bf[i];\n    }\n"
    text = program(body)
    assert_flow(read({"main.cpp": text}), text)


def test_a_device_read_into_the_output_after_the_alias_write_clears_it() -> None:
    body = "    float* o = c.data();\n    for (uint32_t i = 0; i < n; ++i) {\n        o[i] = af[i] + bf[i];\n    }\n"
    assert_clear(read({"main.cpp": program(body + READ_INTO_C)}))


def test_a_device_read_after_the_sink_does_not_clear_the_flow() -> None:
    text = program(HOST_TO_C, tail=WRITE_C + READ_INTO_C)
    assert_flow(read({"main.cpp": text}), text)


def test_a_staging_vector_reused_as_the_read_destination_is_clear() -> None:
    body = (
        "    std::vector<float> staging(af, af + n);\n"
        "    distributed::EnqueueWriteMeshBuffer(cq, in_buf, staging, false);\n"
        "    distributed::EnqueueReadMeshBuffer(cq, staging, c_buf, true);\n"
        "    std::copy(staging.begin(), staging.end(), c.begin());\n"
    )
    assert_clear(read({"main.cpp": program(body)}))


def test_a_host_overwrite_after_the_device_read_is_host_compute() -> None:
    text = program(READ_INTO_C + HOST_TO_C)
    assert_flow(read({"main.cpp": text}), text)


READ_BACK_HELPER = """\
static std::vector<float> read_back(distributed::MeshCommandQueue& queue,
                                    std::shared_ptr<distributed::MeshBuffer>& buffer, uint32_t count) {
    std::vector<float> out(count);
    distributed::EnqueueReadMeshBuffer(queue, out, buffer, true);
    return out;
}
"""


def test_a_whole_variable_assignment_from_a_device_read_is_a_strong_update() -> None:
    body = (
        "    std::vector<float> buf(af, af + n);\n"
        "    distributed::EnqueueWriteMeshBuffer(cq, in_buf, buf, false);\n"
        "    buf = read_back(cq, c_buf, n);\n"
        "    c = buf;\n"
    )
    assert_clear(read({"main.cpp": program(body, helpers=READ_BACK_HELPER)}))


def test_a_nested_template_type_still_declares_a_value() -> None:
    # The '>>' that closes two template lists is split, so rows is value-declared and its assignment kills it.
    helpers = (
        "static std::vector<std::vector<float>> read_rows(distributed::MeshCommandQueue& queue,\n"
        "                                                 std::shared_ptr<distributed::MeshBuffer>& buffer,\n"
        "                                                 uint32_t count) {\n"
        "    std::vector<std::vector<float>> rows(1, std::vector<float>(count));\n"
        "    distributed::EnqueueReadMeshBuffer(queue, rows[0], buffer, true);\n"
        "    return rows;\n"
        "}\n"
    )
    body = (
        "    std::vector<std::vector<float>> rows(1, std::vector<float>(af, af + n));\n"
        "    rows = read_rows(cq, c_buf, n);\n"
        "    c = rows[0];\n"
    )
    assert_clear(read({"main.cpp": program(body, helpers=helpers)}))


def test_same_named_locals_in_sibling_blocks_are_separate_variables() -> None:
    body = (
        "    std::vector<float> staged(n);\n"
        "    for (uint32_t i = 0; i < n; ++i) {\n"
        "        float v = af[i] * 0.5f;\n"
        "        staged[i] = v;\n"
        "    }\n"
        "    distributed::EnqueueWriteMeshBuffer(cq, in_buf, staged, false);\n"
        + READ_BACK
        + "    for (uint32_t i = 0; i < n; ++i) {\n"
        "        float v = res[i];\n"
        "        c[i] = v;\n"
        "    }\n"
    )
    assert_clear(read({"main.cpp": program(body)}))


def test_metadata_and_index_arithmetic_reaching_the_output_are_clear() -> None:
    body = (
        "    std::vector<float> a_copy(af, af + n);\n"
        + READ_BACK
        + "    for (uint32_t i = 0; i < n; ++i) {\n"
        "        c[i] = res[i] * static_cast<float>(a_copy.size());\n"
        "    }\n"
        "    c[0] += static_cast<float>(a.count + b.count + a.rank + b.dims[0] + a.nbytes);\n"
        "    c[1] = res[static_cast<uint32_t>(bf[0]) % n];\n"
        "    c.resize(a_copy.size());\n"
    )
    assert_clear(read({"main.cpp": program(body)}))


def test_an_output_shaped_from_input_metadata_is_clear() -> None:
    # dims built from a.count shape the output but carry no input value. An array declarator owns its storage
    # (_cxx_scan's Structure), so `shape[1] = {a.count}` gives shape a value and links it with nothing; were it
    # an alias, the link shape ~ base({a.count}) = a (base ignores the METADATA rule) would make out read as
    # input data. This test fixes that metadata never carries input values.
    body = (
        READ_BACK
        + "    uint64_t shape[1] = {a.count};\n"
        "    std::vector<float> out(shape[0]);\n"
        "    for (uint32_t i = 0; i < n; ++i) {\n"
        "        out[i] = res[i];\n"
        "    }\n"
    )
    tail = (
        '    if (lassi_io_write(argv[3], "c", LASSI_IO_F32, 1u, shape, out.data()) != 0) {\n'
        "        return 1;\n"
        "    }\n"
    )
    assert_clear(read({"main.cpp": program(body, tail=tail)}))


@pytest.mark.parametrize(
    ("declaration", "flagged"),
    [
        ("    std::vector<float>& r = c;\n", True),
        ("    auto& r = c;\n", True),
        ("    std::vector<float> r = c;\n", False),
        ("    auto r = c;\n", False),
    ],
    ids=["reference", "auto-reference", "value-copy", "auto-copy"],
)
def test_reference_declarations_alias_and_value_declarations_copy(declaration: str, flagged: bool) -> None:
    body = declaration + "    for (uint32_t i = 0; i < n; ++i) {\n        r[i] = af[i];\n    }\n"
    text = program(READ_BACK + DEVICE_TO_C + body)
    result = read({"main.cpp": text})
    if flagged:
        assert_flow(result, text)
    else:
        assert_clear(result)


@pytest.mark.parametrize(("element", "flagged"), [("float& v", True), ("float v", False)], ids=["reference", "value"])
def test_a_range_for_over_the_output_aliases_only_by_reference(element: str, flagged: bool) -> None:
    body = READ_BACK + DEVICE_TO_C + f"    for ({element} : c) {{\n        v = af[0];\n    }}\n"
    text = program(body)
    result = read({"main.cpp": text})
    if flagged:
        assert_flow(result, text)
    else:
        assert_clear(result)


def test_an_initializer_brace_is_not_a_block() -> None:
    text = program("    std::vector<float> w{af[0], af[1]};\n    c.assign(w.begin(), w.end());\n")
    assert_flow(read({"main.cpp": text}), text)


def test_each_declarator_of_a_declaration_has_its_own_value() -> None:
    body = READ_BACK + DEVICE_TO_C + "    float x = af[0], y = 1.0f;\n    (void)x;\n    c[0] = y;\n"
    assert_clear(read({"main.cpp": program(body)}))


def test_a_return_inside_a_lambda_is_not_the_functions_return() -> None:
    helpers = (
        "static float clean_value(const float* x) {\n"
        "    auto first = [&]() { return x[0]; };\n"
        "    (void)first;\n"
        "    return 1.0f;\n"
        "}\n"
    )
    body = READ_BACK + "    for (uint32_t i = 0; i < n; ++i) {\n        c[i] = res[i] * clean_value(af);\n    }\n"
    assert_clear(read({"main.cpp": program(body, helpers=helpers)}))


def test_a_local_shadows_a_file_scope_variable_of_the_same_name() -> None:
    helpers = (
        "static std::vector<float> staged;\n"
        "static void emit_staged(float* out, uint32_t count) {\n"
        "    for (uint32_t i = 0; i < count; ++i) {\n"
        "        out[i] = staged[i];\n"
        "    }\n"
        "}\n"
    )
    body = (
        "    std::vector<float> staged(af, af + n);\n"
        "    distributed::EnqueueWriteMeshBuffer(cq, in_buf, staged, false);\n"
        "    emit_staged(c.data(), n);\n"
    )
    assert_clear(read({"main.cpp": program(body, helpers=helpers)}))


def test_the_witness_names_the_first_violating_sink() -> None:
    tail = (
        "    uint64_t dims[1] = {n};\n"
        '    lassi_io_write(argv[3], "clean", LASSI_IO_F32, 1u, dims, res.data());\n'
        '    lassi_io_write(argv[3], "first", LASSI_IO_F32, 1u, dims, c.data());\n'
        '    lassi_io_write(argv[3], "second", LASSI_IO_F32, 1u, dims, c.data());\n'
    )
    text = program(READ_BACK + HOST_TO_C, tail=tail)
    assert_flow(read({"main.cpp": text}), text, sink='"first"')


def test_a_device_read_kills_only_its_buffer_operands() -> None:
    # Naming the input pointer in the blocking flag does not clear the input.
    body = (
        "    distributed::EnqueueReadMeshBuffer(cq, res, c_buf, af != nullptr);\n"
        "    for (uint32_t i = 0; i < n; ++i) {\n"
        "        c[i] = af[i] * 2.0f;\n"
        "    }\n"
    )
    text = program(body)
    assert_flow(read({"main.cpp": text}), text)


ALIAS_LOOP = "    for (uint32_t i = 0; i < n; ++i) {{\n        {target} = af[i] * 2.0f;\n    }}\n"


@pytest.mark.parametrize(
    ("declaration", "target"),
    [
        ("    auto out = c.data();\n", "out[i]"),
        ("    auto out = &c[0];\n", "out[i]"),
        ("    auto out = c.data() + 0;\n", "out[i]"),
        ("    auto out = reinterpret_cast<float*>(c.data());\n", "out[i]"),
        ("    auto it = c.begin();\n", "*it++"),
        ("    std::vector<float>::iterator it = c.begin();\n", "it[i]"),
        ("    auto it = std::begin(c);\n", "it[i]"),
        ("    std::span<float> out(c);\n", "out[i]"),
        ("    float* __restrict out = c.data();\n", "out[i]"),
    ],
    ids=["auto-data", "auto-address", "auto-offset", "auto-cast", "auto-iterator", "iterator-type",
         "std-begin", "span", "restrict"],
)
def test_pointer_iterator_and_view_declarations_alias_the_output(declaration: str, target: str) -> None:
    text = program(declaration + ALIAS_LOOP.format(target=target))
    assert_flow(read({"main.cpp": text}), text)


@pytest.mark.parametrize("declaration", ["    auto out = c.data();\n", "    std::span<float> out(c);\n"])
def test_an_alias_of_the_output_filled_from_the_device_result_is_clear(declaration: str) -> None:
    body = READ_BACK + declaration + "    for (uint32_t i = 0; i < n; ++i) {\n        out[i] = res[i];\n    }\n"
    assert_clear(read({"main.cpp": program(body)}))


def test_a_file_scope_variable_with_a_parenthesized_initializer_is_a_file_scope_variable() -> None:
    helpers = (
        "std::vector<float> g_out(1024);\n"
        "static void compute(const float* x, uint32_t count) {\n"
        "    for (uint32_t i = 0; i < count; ++i) {\n"
        "        g_out[i] = x[i] * 2.0f;\n"
        "    }\n"
        "}\n"
    )
    tail = '    uint64_t dims[1] = {n};\n    lassi_io_write(argv[3], "c", LASSI_IO_F32, 1u, dims, g_out.data());\n'
    text = program("    compute(af, n);\n", helpers=helpers, tail=tail)
    assert_flow(read({"main.cpp": text}), text)


@pytest.mark.parametrize(
    ("helpers", "name"), [("namespace host {\nfloat out[64];\n}\n", "host::out"), ("float g_out[64];\n", "::g_out")]
)
def test_a_qualified_name_of_a_file_scope_variable_resolves_to_it(helpers: str, name: str) -> None:
    body = f"    for (uint32_t i = 0; i < 64; ++i) {{\n        {name}[i] = af[i] * 2.0f;\n    }}\n"
    tail = f'    uint64_t dims[1] = {{64}};\n    lassi_io_write(argv[3], "c", LASSI_IO_F32, 1u, dims, {name});\n'
    text = program(body, helpers=helpers, tail=tail)
    assert_flow(read({"main.cpp": text}), text)


@pytest.mark.parametrize("binding", ["auto [out, status]", "const auto& [out, status]"])
def test_a_structured_binding_declares_each_name(binding: str) -> None:
    helpers = (
        "static std::pair<std::vector<float>, int> twice(const float* x, uint32_t count) {\n"
        "    std::vector<float> o(count);\n"
        "    for (uint32_t i = 0; i < count; ++i) {\n"
        "        o[i] = 2.0f * x[i];\n"
        "    }\n"
        "    return {o, 0};\n"
        "}\n"
    )
    text = program(f"    {binding} = twice(af, n);\n    (void)status;\n", helpers=helpers, tail=WRITE_OUT)
    assert_flow(read({"main.cpp": text}), text)


LOAD_PAIRS = """\
static std::pair<std::vector<float>, uint32_t> load_pair(const float* p, uint32_t count) {
    return {std::vector<float>(p, p + count), count};
}
static std::vector<std::pair<std::vector<float>, uint32_t>> load_pairs(const float* p, uint32_t count) {
    return {load_pair(p, count)};
}
"""


@pytest.mark.parametrize(
    "body",
    [
        "    auto&& [vals, cnt] = load_pair(af, n);\n"
        "    (void)cnt;\n"
        "    distributed::EnqueueWriteMeshBuffer(cq, in_buf, vals, false);\n"
        + READ_BACK
        + "    vals = res;\n"
        "    c = vals;\n",
        READ_BACK
        + "    for (auto&& [vals, cnt] : load_pairs(af, n)) {\n"
        "        (void)cnt;\n"
        "        vals = res;\n"
        "        c = vals;\n"
        "    }\n",
    ],
    ids=["declaration", "range-for"],
)
def test_a_whole_assignment_to_a_reference_binding_name_of_a_call_replaces_its_value(body: str) -> None:
    # A reference binding of a call has no base, so each name holds the call's value as a value binding's does, and
    # `vals = res` is a strong update of it, as for a value-declared name (the P4.12 commit audit). Without the
    # assignment the input reaches c.
    assert_clear(read({"main.cpp": program(body, helpers=LOAD_PAIRS)}))
    text = program(body.replace("vals = res;", "(void)res;"), helpers=LOAD_PAIRS)
    assert_flow(read({"main.cpp": text}), text)


PAIRS_FROM_INPUTS = (
    "    std::vector<std::pair<float, float>> pairs;\n"
    "    for (uint32_t i = 0; i < n; ++i) {\n"
    "        pairs.emplace_back(af[i], bf[i]);\n"
    "    }\n"
    "    c.clear();\n"
)


@pytest.mark.parametrize(
    "header",
    [
        "const auto& [x, y] : pairs",
        "auto [x, y] : pairs",
        "auto&& [x, y] : pairs",
        "auto& p = pairs; const auto& [x, y] : p",
    ],
    ids=["const-reference", "value", "forwarding-reference", "init-statement"],
)
def test_a_range_for_structured_binding_gives_each_name_the_ranges_values(header: str) -> None:
    # A value binding defines each name from the range; a reference binding links each name with it.
    text = program(PAIRS_FROM_INPUTS + f"    for ({header}) {{\n        c.push_back(x + y);\n    }}\n")
    assert_flow(read({"main.cpp": text}), text)


# Honest programs that use, inside a construct, a name the construct declares, beside an input-holding variable
# of the same name outside it. Each rename changes only the declared name, so the use then names the outer
# variable and reads True.
SHADOWING_BINDING = (
    "    float value = af[0];\n"
    "    (void)value;\n"
    "    std::vector<std::pair<float, float>> pairs(n);\n"
    + READ_BACK
    + "    for (uint32_t i = 0; i < n; ++i) {\n"
    "        pairs[i] = {res[i], 0.0f};\n"
    "    }\n"
    "    c.clear();\n"
    "    for (const auto& [value, bias] : pairs) {\n"
    "        c.push_back(value + bias);\n"
    "    }\n"
)
SHADOWED = [
    (SHADOWING_BINDING, ("[value, bias]", "[sum, bias]")),
    (
        SHADOWING_BINDING.replace("for (const auto&", "for (auto& p = pairs; const auto&").replace(
            "bias] : pairs", "bias] : p"
        ),
        ("[value, bias]", "[sum, bias]"),
    ),
    (
        READ_BACK
        + "    [&, af = res.data()]() {\n"
        "        for (uint32_t i = 0; i < n; ++i) {\n"
        "            c[i] = af[i];\n"
        "        }\n"
        "    }();\n",
        ("af = res.data()", "r = res.data()"),
    ),
    (
        "    float e = af[0];\n"
        "    (void)e;\n"
        + READ_BACK
        + DEVICE_TO_C
        + "    try {\n        throw 1.0f;\n    } catch (float e) {\n        c[0] = e;\n    }\n",
        ("catch (float e)", "catch (float g)"),
    ),
]


@pytest.mark.parametrize(
    ("body", "rename"),
    SHADOWED,
    ids=["range-for-binding", "range-for-binding-after-an-init-statement", "init-capture", "catch-parameter"],
)
def test_a_range_for_binding_an_init_capture_and_a_catch_parameter_hide_an_outer_name(
    body: str, rename: tuple[str, str]
) -> None:
    assert_clear(read({"main.cpp": program(body)}))
    text = program(body.replace(*rename))
    assert_flow(read({"main.cpp": text}), text)


# A header that declares `value` from device data, after an input-holding outer variable of the same name. Each
# (header, controlled statement) pair is read unbraced and braced: inside the statement `value` is the header's
# declaration, and after it the outer variable again (the unbraced forms were found by the P4.12 commit audit).
HEADER_PAIRS = (
    "    std::vector<std::pair<float, float>> pairs(n);\n"
    + READ_BACK
    + "    for (uint32_t i = 0; i < n; ++i) {\n"
    "        pairs[i] = {res[i], 0.0f};\n"
    "    }\n"
    "    float value = af[0];\n"
)
HEADER_STATEMENTS = [
    ("for (const auto& [value, bias] : pairs)", "c[1] = value + bias;"),
    ("for (const auto& value : res)", "c[1] = value;"),
    ("for (float value = res[0]; value < 0.0f;)", "break;"),
    ("while (const float value = res[0])", "break;"),
    ("if (const float value = res[0]; value > 0.0f)", "c[1] = value;"),
    ("switch (const int value = static_cast<int>(res[0]))", "default: c[1] = static_cast<float>(value);"),
]
USE_OUTER_VALUE = "    c[0] = value;\n"


@pytest.mark.parametrize("braced", [False, True], ids=["unbraced", "braced"])
@pytest.mark.parametrize(
    ("header", "controlled"),
    HEADER_STATEMENTS,
    ids=["range-for-binding", "range-for-element", "for-init", "while-condition", "if-init", "switch-condition"],
)
def test_a_header_declaration_belongs_to_the_statement_the_header_controls(
    header: str, controlled: str, braced: bool
) -> None:
    statement = f"    {header} {{\n        {controlled}\n    }}\n" if braced else f"    {header} {controlled}\n"
    assert_clear(read({"main.cpp": program(HEADER_PAIRS + statement)}))
    text = program(HEADER_PAIRS + statement + USE_OUTER_VALUE)
    assert_flow(read({"main.cpp": text}), text)


# Controlled statements that do not end at their first `;`: a try statement ends at its last handler's `}`, and an
# attribute `[[...]]` leads the statement it belongs to (the P4.12 commit audit found both read to the next `;`).
TRY_AND_ATTRIBUTE_STATEMENTS = [
    "try {\n        c[1] = value;\n    } catch (...) {\n    }\n",
    "try {\n        c[1] = value;\n    } catch (const std::exception& e) {\n        (void)e;\n"
    "    } catch (...) {\n    }\n",
    "[[likely]] {\n        c[1] = value;\n    }\n",
    "[[likely]] c[1] = value;\n",
]
TRY_AND_ATTRIBUTE_IDS = ["try", "try-with-two-handlers", "attribute-block", "attribute-statement"]


@pytest.mark.parametrize("controlled", TRY_AND_ATTRIBUTE_STATEMENTS, ids=TRY_AND_ATTRIBUTE_IDS)
@pytest.mark.parametrize(
    "header", ["if (const float value = res[0])", "for (const auto& value : res)"], ids=["if-condition", "range-for"]
)
def test_a_header_declaration_ends_with_a_controlled_try_or_attribute_led_statement(
    header: str, controlled: str
) -> None:
    statement = f"    {header} {controlled}"
    assert_clear(read({"main.cpp": program(HEADER_PAIRS + statement)}))
    text = program(HEADER_PAIRS + statement + USE_OUTER_VALUE)
    assert_flow(read({"main.cpp": text}), text)


@pytest.mark.parametrize("controlled", TRY_AND_ATTRIBUTE_STATEMENTS, ids=TRY_AND_ATTRIBUTE_IDS)
def test_a_loop_whose_body_is_a_try_or_attribute_led_statement_ends_with_it(controlled: str) -> None:
    # The staging write after the loop lies outside the loop, so no back edge carries it to the read in the body;
    # inside a braced body it does.
    body = controlled.replace("c[1] = value;", "c[i] = res[i];")
    after = READ_BACK + f"    for (uint32_t i = 0; i < n; ++i) {body}" + "    res[0] = af[0];\n"
    assert_clear(read({"main.cpp": program(after)}))
    inside = READ_BACK + "    for (uint32_t i = 0; i < n; ++i) {\n" + f"    {body}" + "        res[0] = af[0];\n    }\n"
    text = program(inside)
    assert_flow(read({"main.cpp": text}), text)


# An input-holding v, then a block led by an attribute that declares its own v from the device result.
OUTER_V = "    float v = af[0];\n    (void)v;\n"
INNER_V = "        float v = res[0];\n"
ATTRIBUTE_BLOCKS = [
    "    for (uint32_t i = 0; i < n; ++i) [[likely]] {\n" + INNER_V + "        c[i] = v;\n    }\n",
    "    if (n > 0) [[likely]] {\n" + INNER_V + "        c[0] = v;\n    }\n",
    "    [[likely]] {\n" + INNER_V + "        c[0] = v;\n    }\n",
]


@pytest.mark.parametrize("block", ATTRIBUTE_BLOCKS, ids=["loop-body", "if-branch", "statement"])
def test_a_block_led_by_an_attribute_is_a_block(block: str) -> None:
    # Found by the P4.12 commit audit: the block was read as part of one statement running to the next `;`, so its
    # declaration was lost and v named the outer variable. Without the inner declaration it does.
    assert_clear(read({"main.cpp": program(OUTER_V + READ_BACK + block)}))
    text = program(OUTER_V + READ_BACK + block.replace(INNER_V, ""))
    assert_flow(read({"main.cpp": text}), text)


UNLIKELY_RETURN = "    if (n == 0) [[unlikely]] {\n        return 1;\n    }\n"


def test_the_statement_after_an_attribute_led_block_is_its_own_statement() -> None:
    # Found by the P4.12 commit audit: the statement after the block was read as part of it, so its declarations, a
    # named lambda's included, were lost. Here the fetch lambda refills staged, and a plain copy of staged reads True.
    refill = (
        "    auto fetch = [&](std::vector<float>& v) {\n"
        "        distributed::EnqueueReadMeshBuffer(cq, v, c_buf, true);\n"
        "    };\n"
        "    std::vector<float> staged(af, af + n);\n"
        "    distributed::EnqueueWriteMeshBuffer(cq, in_buf, staged, false);\n"
        "    fetch(staged);\n"
        "    c = staged;\n"
    )
    assert_clear(read({"main.cpp": program(UNLIKELY_RETURN + refill)}))
    text = program(UNLIKELY_RETURN + "    std::vector<float> staged(af, af + n);\n    c = staged;\n")
    assert_flow(read({"main.cpp": text}), text)


@pytest.mark.parametrize("braced", [False, True], ids=["unbraced", "braced"])
def test_an_if_condition_declaration_is_seen_in_its_else_branch(braced: bool) -> None:
    if braced:
        branches = " {\n        c[0] = value;\n    } else {\n        c[0] = -value;\n    }\n"
    else:
        branches = " c[0] = value;\n    else c[0] = -value;\n"
    statement = "    if (const float value = res[0]; value > 0.0f)" + branches
    assert_clear(read({"main.cpp": program(HEADER_PAIRS + statement)}))
    text = program(HEADER_PAIRS + statement + "    c[1] = value;\n")
    assert_flow(read({"main.cpp": text}), text)


# A condition declares a name only as C++ allows, with an `=` or braced initializer: `use_tt && af`, `mask & v`, and
# `scale * x` are expressions, so the second name still names the input-holding variable in the statement the
# header controls (the P4.12 commit audit). Each pair is (condition, a use of that name in a branch).
TWO_NAME_SCALARS = (
    "    const bool use_tt = n > 0;\n"
    "    const uint32_t mask = 1u;\n"
    "    const uint32_t v = static_cast<uint32_t>(af[0]);\n"
    "    const float scale = 2.0f;\n"
    "    const float x = af[1];\n"
)
TWO_NAME_CONDITIONS = [
    ("use_tt && af", "c[0] = af[0];"),
    ("mask & v", "c[0] = static_cast<float>(v);"),
    ("scale * x", "c[0] = x;"),
]


@pytest.mark.parametrize("braced", [False, True], ids=["unbraced", "braced"])
@pytest.mark.parametrize("branch", ["then", "else"])
@pytest.mark.parametrize(("condition", "use"), TWO_NAME_CONDITIONS, ids=["and", "bitand", "times"])
def test_an_if_condition_with_no_initializer_declares_nothing(
    condition: str, use: str, branch: str, braced: bool
) -> None:
    first, second = (use, "c[0] = 0.0f;") if branch == "then" else ("c[0] = 0.0f;", use)
    if braced:
        statement = f"    if ({condition}) {{\n        {first}\n    }} else {{\n        {second}\n    }}\n"
    else:
        statement = f"    if ({condition}) {first}\n    else {second}\n"
    text = program(TWO_NAME_SCALARS + statement)
    assert_flow(read({"main.cpp": text}), text)
    assert_clear(read({"main.cpp": program(TWO_NAME_SCALARS + statement.replace(use, "c[0] = 1.0f;"))}))


@pytest.mark.parametrize(
    "statement",
    [
        "    while (use_tt && af) {\n        c[0] = af[0];\n        break;\n    }\n",
        "    switch (mask & v) {\n        default:\n            c[0] = static_cast<float>(v);\n    }\n",
        "    for (uint32_t i = 0; use_tt && af; ++i) {\n        c[i] = af[i];\n        break;\n    }\n",
        "    for (uint32_t i = 0; i < 1u; scale * x) {\n        c[i] = x;\n        break;\n    }\n",
    ],
    ids=["while", "switch", "for-condition", "for-increment"],
)
def test_a_while_switch_or_for_header_expression_declares_nothing(statement: str) -> None:
    # A while or switch condition and a for's second part follow the if condition's rule; a for's third part is an
    # expression, where C++ allows no declaration.
    text = program(TWO_NAME_SCALARS + statement)
    assert_flow(read({"main.cpp": text}), text)


def test_a_call_in_a_condition_is_no_declaration() -> None:
    # `use_tt && fill(c)` has the form of a declaration of fill with a parenthesized initializer, which C++ does not
    # allow in a condition: it is a call of the named lambda fill, which writes c from the input.
    body = (
        "    const bool use_tt = n > 0;\n"
        "    auto fill = [&](std::vector<float>& out) {\n"
        "        for (uint32_t i = 0; i < n; ++i) {\n"
        "            out[i] = af[i];\n"
        "        }\n"
        "        return true;\n"
        "    };\n"
        "    if (use_tt && fill(c)) {\n"
        "        c[0] += 0.0f;\n"
        "    }\n"
    )
    text = program(body)
    assert_flow(read({"main.cpp": text}), text)


# A reference-declared structured binding (in a declaration or a range-for header) and a reference init capture (of
# a lambda that is not a named lambda called inside another) are alias links of base(E), as a reference declaration
# is: a write through the name reaches the bound object, and a device read through it clears the object. A value
# binding and a value init capture hold copies.
REFERENCE_NAME_WRITES = [
    (
        "    std::pair<std::vector<float>, int> p{std::vector<float>(n), 0};\n"
        "    auto& [values, status] = p;\n"
        "    (void)status;\n"
        "    for (uint32_t i = 0; i < n; ++i) {\n"
        "        values[i] = af[i];\n"
        "    }\n"
        "    c = p.first;\n",
        ("auto& [values", "auto [values"),
    ),
    (
        "    std::vector<std::pair<float, float>> pairs(n);\n"
        "    for (auto& [x, y] : pairs) {\n"
        "        x = af[0];\n"
        "        y = bf[0];\n"
        "    }\n"
        "    for (uint32_t i = 0; i < n; ++i) {\n"
        "        c[i] = pairs[i].first + pairs[i].second;\n"
        "    }\n",
        ("auto& [x, y]", "auto [x, y]"),
    ),
    (
        READ_BACK + DEVICE_TO_C + "    [&r = c]() {\n        r[0] = af[0];\n    }();\n",
        ("[&r = c]()", "[r = c]() mutable"),
    ),
]


@pytest.mark.parametrize(
    ("body", "copy"), REFERENCE_NAME_WRITES, ids=["declaration", "range-for", "reference-init-capture"]
)
def test_a_write_through_a_reference_binding_or_init_capture_reaches_the_bound_object(
    body: str, copy: tuple[str, str]
) -> None:
    # These three bodies were documented misses (test_limit_*) while binding names and init captures had no link.
    text = program(body)
    assert_flow(read({"main.cpp": text}), text)
    assert_clear(read({"main.cpp": program(body.replace(*copy))}))


DEVICE_READ_THROUGH_A_BINDING = (
    "    std::pair<std::vector<float>, int> out{std::vector<float>(af, af + n), 0};\n"
    "    auto& [buf, k] = out;\n"
    "    (void)k;\n"
    "    distributed::EnqueueReadMeshBuffer(cq, buf, c_buf, true);\n"
    "    c = out.first;\n"
)
DEVICE_READ_THROUGH_A_RANGE_FOR_BINDING = (
    "    std::vector<std::pair<std::vector<float>, int>> outs;\n"
    "    outs.emplace_back(std::vector<float>(af, af + n), 0);\n"
    "    for (auto& [buf, k] : outs) {\n"
    "        (void)k;\n"
    "        distributed::EnqueueReadMeshBuffer(cq, buf, c_buf, true);\n"
    "    }\n"
    "    c = outs[0].first;\n"
)
DEVICE_READ_THROUGH_AN_INIT_CAPTURE = (
    "    std::vector<float> staged(af, af + n);\n"
    "    [&r = staged, &cq, &c_buf]() {\n"
    "        distributed::EnqueueReadMeshBuffer(cq, r, c_buf, true);\n"
    "    }();\n"
    "    c = staged;\n"
)
# (body, the control that names the object directly, the body with the device read filling another buffer)
DEVICE_READS_THROUGH = [
    (
        DEVICE_READ_THROUGH_A_BINDING,
        DEVICE_READ_THROUGH_A_BINDING.replace("(cq, buf,", "(cq, out.first,"),
        DEVICE_READ_THROUGH_A_BINDING.replace("(cq, buf,", "(cq, res,"),
    ),
    (
        DEVICE_READ_THROUGH_A_RANGE_FOR_BINDING,
        "    std::vector<std::vector<float>> outs;\n"
        "    outs.emplace_back(af, af + n);\n"
        "    for (auto& buf : outs) {\n"
        "        distributed::EnqueueReadMeshBuffer(cq, buf, c_buf, true);\n"
        "    }\n"
        "    c = outs[0];\n",
        DEVICE_READ_THROUGH_A_RANGE_FOR_BINDING.replace("(cq, buf,", "(cq, res,"),
    ),
    (
        DEVICE_READ_THROUGH_AN_INIT_CAPTURE,
        DEVICE_READ_THROUGH_AN_INIT_CAPTURE.replace("[&r = staged, &cq, &c_buf]", "[&]").replace(
            "(cq, r,", "(cq, staged,"
        ),
        DEVICE_READ_THROUGH_AN_INIT_CAPTURE.replace("[&r = staged,", "[&r = res,"),
    ),
]


@pytest.mark.parametrize(
    ("body", "named", "elsewhere"),
    DEVICE_READS_THROUGH,
    ids=["declared-binding", "range-for-binding", "reference-init-capture"],
)
def test_a_device_read_through_a_reference_binding_or_init_capture_clears_the_object(
    body: str, named: str, elsewhere: str
) -> None:
    # Found by the P4.12 commit audit: each object is filled from the input and refilled by a device read through
    # the name. Naming the object directly reads the same; refilling another buffer leaves the input in it.
    assert_clear(read({"main.cpp": program(body)}))
    assert_clear(read({"main.cpp": program(named)}))
    text = program(elsewhere)
    assert_flow(read({"main.cpp": text}), text)


RUNNER_WITH_EXPR = """\
#define SCALE 1.0f
class Runner {
  public:
    void load(const float* x, uint32_t count) {
        stage.assign(x, x + count);
    }
    void finish(const std::vector<float>& from, std::vector<float>& to) const {
        for (std::size_t i = 0; i < from.size(); ++i) {
            to[i] = from[i] * EXPR;
        }
    }

  private:
    std::vector<float> stage;
};
"""


@pytest.mark.parametrize("expression", ["1.0f", "SCALE", "FLT_MAX"])
def test_a_name_that_is_no_data_member_does_not_read_the_object(expression: str) -> None:
    # In a member function only the class's data members are THIS: a macro or an undeclared name is not.
    helpers = RUNNER_WITH_EXPR.replace("EXPR", expression)
    body = "    Runner r;\n    r.load(af, n);\n" + READ_BACK + "    r.finish(res, c);\n"
    assert_clear(read({"main.cpp": program(body, helpers=helpers)}))


def test_in_a_class_with_an_unknown_base_an_undeclared_name_is_a_member() -> None:
    helpers = (
        "struct Loader : BaseLoader {\n"
        "    void keep(const float* x, uint32_t count) {\n"
        "        staged.assign(x, x + count);\n"
        "    }\n"
        "    void emit(float* out, uint32_t count) const {\n"
        "        for (uint32_t i = 0; i < count; ++i) {\n"
        "            out[i] = staged[i];\n"
        "        }\n"
        "    }\n"
        "};\n"
    )
    text = program("    Loader loader;\n    loader.keep(af, n);\n    loader.emit(c.data(), n);\n", helpers=helpers)
    assert_flow(read({"main.cpp": text}), text)


# ---------------------------------------------------------------------------
# Counts and metadata: an output's size carries no value


OUT_FROM_DEVICE = READ_BACK + "    for (uint32_t i = 0; i < n; ++i) {\n        out[i] = res[i];\n    }\n"
ROWS_FROM_INPUT = "    const uint32_t rows = static_cast<uint32_t>(af[0]);\n"


@pytest.mark.parametrize(
    ("helpers", "sizing"),
    [
        ("", "    const uint64_t* shape = a.dims;\n    std::vector<float> out(static_cast<uint32_t>(shape[0]));\n"),
        ("", ROWS_FROM_INPUT + "    std::vector<float> out(rows);\n"),
        ("", ROWS_FROM_INPUT + "    std::vector<float> out(rows, 0.0f);\n"),
        ("", ROWS_FROM_INPUT + "    auto out = std::vector<float>(rows);\n"),
        ("", ROWS_FROM_INPUT + "    std::vector<float> out;\n    out.assign(rows, 0.0f);\n"),
        ("", ROWS_FROM_INPUT + "    std::vector<float> out;\n    out.insert(out.end(), rows, 0.0f);\n"),
        ("", "    std::vector<float> in_a(af, af + n);\n    std::vector<float> out(std::size(in_a));\n"),
        ("", "    const auto rows = std::distance(af, af + n);\n    std::vector<float> out(rows);\n"),
        (
            "struct Matrix {\n    uint32_t rows;\n    std::vector<float> data;\n};\n",
            "    Matrix m{static_cast<uint32_t>(a.dims[0]), std::vector<float>(af, af + n)};\n"
            "    std::vector<float> out(m.rows);\n",
        ),
    ],
    ids=["dims-pointer", "count", "count-and-value", "count-temporary", "assign-count", "insert-count", "std-size",
         "distance", "struct-field"],
)
def test_a_count_sizes_the_output_but_carries_no_value(helpers: str, sizing: str) -> None:
    assert_clear(read({"main.cpp": program(sizing + OUT_FROM_DEVICE, helpers=helpers, tail=WRITE_OUT)}))


def test_a_sized_declaration_and_a_resize_read_alike() -> None:
    sized = ROWS_FROM_INPUT + "    std::vector<float> out(rows);\n"
    resized = ROWS_FROM_INPUT + "    std::vector<float> out;\n    out.resize(rows);\n"
    first, second = (read({"main.cpp": program(item + OUT_FROM_DEVICE, tail=WRITE_OUT)}) for item in (sized, resized))
    assert first == second and first.host_compute is False


@pytest.mark.parametrize(
    "filling",
    [
        "    std::vector<float> out(af, af + n);\n",
        "    std::vector<float> in_a(af, af + n);\n    std::vector<float> out(in_a);\n",
        "    auto in_a = std::vector<float>(af, af + n);\n    std::vector<float> out(in_a);\n",
        "    std::vector<float> out(n, af[0]);\n",
        "    std::vector<float> out;\n    out.assign(af, af + n);\n",
        "    std::vector<float> out;\n    out.assign(n, af[0]);\n",
    ],
    ids=["range", "copy", "auto-copy", "count-and-input-value", "assign-range", "assign-count-and-input-value"],
)
def test_ranges_copies_and_fill_values_still_carry_values(filling: str) -> None:
    text = program(filling, tail=WRITE_OUT)
    assert_flow(read({"main.cpp": text}), text)


# ---------------------------------------------------------------------------
# Visibility: loops, branches, and lambdas


def test_a_staging_vector_reused_for_two_outputs_is_clear() -> None:
    # A later pass's staging write is not a back edge to an earlier read: no loop holds both.
    body = (
        "    std::vector<float> stage(n);\n"
        "    for (uint32_t i = 0; i < n; ++i) {\n"
        "        stage[i] = af[i];\n"
        "    }\n"
        "    distributed::EnqueueWriteMeshBuffer(cq, in_buf, stage, false);\n"
        "    distributed::EnqueueReadMeshBuffer(cq, stage, c_buf, true);\n"
        "    c = stage;\n"
        "    uint64_t dims[1] = {n};\n"
        '    lassi_io_write(argv[3], "c", LASSI_IO_F32, 1u, dims, c.data());\n'
        "    for (uint32_t i = 0; i < n; ++i) {\n"
        "        stage[i] = bf[i];\n"
        "    }\n"
        "    distributed::EnqueueWriteMeshBuffer(cq, in_buf, stage, false);\n"
        "    distributed::EnqueueReadMeshBuffer(cq, stage, c_buf, true);\n"
        "    std::vector<float> d(stage);\n"
        '    lassi_io_write(argv[3], "d", LASSI_IO_F32, 1u, dims, d.data());\n'
    )
    assert_clear(read({"main.cpp": program(body, tail="")}))


def test_a_scalar_reused_after_its_strong_update_is_clear() -> None:
    body = READ_BACK + (
        "    float t = 0.0f;\n"
        "    for (uint32_t i = 0; i < n; ++i) {\n"
        "        t = res[i];\n"
        "        c[i] = t;\n"
        "    }\n"
    )
    tail = WRITE_C + (
        "    int bad = 0;\n"
        "    for (uint32_t i = 0; i < n; ++i) {\n"
        "        t = af[i] + bf[i];\n"
        "        if (c[i] != t) {\n"
        "            ++bad;\n"
        "        }\n"
        "    }\n"
    )
    assert_clear(read({"main.cpp": program(body, tail=tail)}))


def test_a_loop_carried_host_flow_is_host_compute() -> None:
    body = (
        "    std::vector<float> carry(n);\n"
        "    for (int pass = 0; pass < 2; ++pass) {\n"
        "        for (uint32_t i = 0; i < n; ++i) {\n"
        "            c[i] = carry[i];\n"
        "        }\n"
        "        for (uint32_t i = 0; i < n; ++i) {\n"
        "            carry[i] = af[i];\n"
        "        }\n"
        "    }\n"
    )
    text = program(body)
    assert_flow(read({"main.cpp": text}), text)


@pytest.mark.parametrize("where", ["top-of-loop", "end-of-loop"])
def test_a_device_read_in_the_loop_cuts_the_back_edge(where: str) -> None:
    refill = "        distributed::EnqueueReadMeshBuffer(cq, buf, c_buf, true);\n"
    body = (
        "    std::vector<float> buf(n);\n"
        "    for (int pass = 0; pass < 2; ++pass) {\n"
        + (refill if where == "top-of-loop" else "")
        + "        for (uint32_t i = 0; i < n; ++i) {\n"
        "            c[i] = buf[i];\n"
        "        }\n"
        "        for (uint32_t i = 0; i < n; ++i) {\n"
        "            buf[i] = af[i];\n"
        "        }\n"
        "        distributed::EnqueueWriteMeshBuffer(cq, in_buf, buf, false);\n"
        + (refill if where == "end-of-loop" else "")
        + "    }\n"
    )
    assert_clear(read({"main.cpp": program(body)}))


STAGE_ROUND = """\
        for (uint32_t i = 0; i < n; ++i) {{
            host[i] = {source}[i];
        }}
        distributed::EnqueueWriteMeshBuffer(cq, in_buf, host, false);
        distributed::EnqueueReadMeshBuffer(cq, host, c_buf, true);
        for (uint32_t i = 0; i < n; ++i) {{
            c[i] = host[i];
        }}
"""


@pytest.mark.parametrize("form", ["if-else", "switch"])
def test_staging_rounds_in_sibling_branches_are_clear(form: str) -> None:
    first, second = STAGE_ROUND.format(source="af"), STAGE_ROUND.format(source="bf")
    if form == "if-else":
        branches = "    if (a.dtype == LASSI_IO_F32) {\n" + first + "    } else {\n" + second + "    }\n"
    else:
        branches = (
            "    switch (a.dtype) {\n    case LASSI_IO_F32: {\n" + first + "        break;\n    }\n"
            "    default: {\n" + second + "        break;\n    }\n    }\n"
        )
    assert_clear(read({"main.cpp": program("    std::vector<float> host(n);\n" + branches)}))


def test_a_named_lambda_runs_where_it_is_called() -> None:
    # The conversion lambda is defined before the read-back but called after it, so it reads device data.
    convert = (
        "    std::vector<float> host(n);\n"
        "    for (uint32_t i = 0; i < n; ++i) {\n"
        "        host[i] = af[i];\n"
        "    }\n"
        "    distributed::EnqueueWriteMeshBuffer(cq, in_buf, host, false);\n"
        "    auto unpack = [&]() {\n"
        "        for (uint32_t i = 0; i < n; ++i) {\n"
        "            c[i] = host[i] * 0.5f;\n"
        "        }\n"
        "    };\n"
        "    distributed::EnqueueReadMeshBuffer(cq, host, c_buf, true);\n"
        "    unpack();\n"
    )
    assert_clear(read({"main.cpp": program(convert)}))
    fetch = (
        "    std::vector<float> host(n);\n"
        "    auto fetch = [&]() {\n"
        "        distributed::EnqueueReadMeshBuffer(cq, host, c_buf, true);\n"
        "        for (uint32_t i = 0; i < n; ++i) {\n"
        "            c[i] = host[i];\n"
        "        }\n"
        "    };\n"
        "    for (uint32_t i = 0; i < n; ++i) {\n"
        "        host[i] = af[i];\n"
        "    }\n"
        "    distributed::EnqueueWriteMeshBuffer(cq, in_buf, host, false);\n"
        "    fetch();\n"
    )
    assert_clear(read({"main.cpp": program(fetch)}))
    compute = (
        "    std::vector<float> scratch(n);\n"
        "    auto twice = [&]() {\n"
        "        for (uint32_t i = 0; i < n; ++i) {\n"
        "            c[i] = scratch[i] * 2.0f;\n"
        "        }\n"
        "    };\n"
        "    for (uint32_t i = 0; i < n; ++i) {\n"
        "        scratch[i] = af[i];\n"
        "    }\n"
        "    twice();\n"
    )
    text = program(compute)
    assert_flow(read({"main.cpp": text}), text)


EMIT_LAMBDA = """\
    auto emit = [&](const char* path, const std::vector<float>& v) {
        uint64_t shape[1] = {v.size()};
        return lassi_io_write(path, "c", LASSI_IO_F32, 1u, shape, v.data());
    };
"""


def test_a_named_lambda_reads_its_arguments_where_it_is_called() -> None:
    text = program(HOST_TO_C + EMIT_LAMBDA + "    emit(argv[3], c);\n", tail="")
    assert_flow(read({"main.cpp": text}), text, sink="lassi_io_write(path")
    assert_clear(read({"main.cpp": program(CLEAN_BODY + EMIT_LAMBDA + "    emit(argv[3], c);\n", tail="")}))


# Named lambdas that do nothing, each named like a standard call, a POSIX read or write, or an own function, and a
# call of each: (lambda, call, the helpers, the input read and the output write the call alone gives).
OWN_COMBINE = VIOLATING_OPS.replace("#pragma once\n", "")
LAMBDAS_NAMED_LIKE_CALLS = [
    (
        "    auto fill = [](std::vector<float>& out, const float* x) {\n        (void)out;\n        (void)x;\n    };\n",
        "    fill(c, af);\n",
        "",
        READ_A,
        WRITE_NEEDLE,
    ),
    (
        "    auto read = [](const char* path, std::vector<float>& v, int k) {\n"
        "        (void)path;\n"
        "        (void)v;\n"
        "        (void)k;\n"
        "    };\n",
        READ_BACK + "    read(argv[1], res, 0);\n" + DEVICE_TO_C,
        "",
        "read(argv[1], res",
        WRITE_NEEDLE,
    ),
    (
        "    auto write = [](int fd, const float* p, size_t k) {\n"
        "        (void)fd;\n"
        "        (void)p;\n"
        "        (void)k;\n"
        "    };\n",
        CLEAN_BODY + "    write(1, af, n);\n",
        "",
        READ_A,
        "write(1, af",
    ),
    (
        "    auto combine = [](const float* x, const float* y, float* z, uint32_t count) {\n"
        "        (void)x;\n"
        "        (void)y;\n"
        "        (void)z;\n"
        "        (void)count;\n"
        "    };\n",
        COMBINE_BODY,
        OWN_COMBINE,
        READ_A,
        WRITE_NEEDLE,
    ),
]


@pytest.mark.parametrize(
    ("lam", "call", "helpers", "source", "sink"),
    LAMBDAS_NAMED_LIKE_CALLS,
    ids=["standard-call", "posix-read", "posix-write", "own-function"],
)
def test_an_unqualified_call_of_a_named_lambda_calls_only_that_lambda(
    lam: str, call: str, helpers: str, source: str, sink: str
) -> None:
    # Found by the P4.12 commit audit: a local name hides every function of its name, so these calls run the lambda,
    # which does nothing, and are not std::fill, POSIX read or write, or the own combine. Without the lambda each
    # call is that function and reads True.
    assert_clear(read({"main.cpp": program(lam + call, helpers=helpers)}))
    text = program(call, helpers=helpers)
    assert_flow(read({"main.cpp": text}), text, source=source, sink=sink)


@pytest.mark.parametrize(
    ("body", "copy"),
    [
        (
            "    auto add = [](const float* x, const float* y, uint32_t count, std::vector<float>& out) {\n"
            "        for (uint32_t i = 0; i < count; ++i) {\n"
            "            out[i] = x[i] + y[i];\n"
            "        }\n"
            "    };\n"
            "    add(af, bf, n, c);\n",
            ("std::vector<float>& out", "std::vector<float> out"),
        ),
        (
            "    auto add = [](const float* x, const float* y, uint32_t count, float* out) {\n"
            "        for (uint32_t i = 0; i < count; ++i) {\n"
            "            out[i] = x[i] + y[i];\n"
            "        }\n"
            "    };\n"
            "    add(af, bf, n, c.data());\n",
            ("add(af, bf, n, c.data())", "add(af, bf, n, res.data())"),
        ),
        (
            "    std::function<void(std::vector<float>&)> add = [&](std::vector<float>& out) {\n"
            "        for (uint32_t i = 0; i < n; ++i) {\n"
            "            out[i] = af[i] + bf[i];\n"
            "        }\n"
            "    };\n"
            "    add(c);\n",
            ("std::vector<float>&", "std::vector<float>"),
        ),
    ],
    ids=["reference-parameter", "pointer-parameter", "std-function"],
)
def test_a_named_lambda_writing_through_a_pointer_or_reference_parameter_writes_its_argument(
    body: str, copy: tuple[str, str]
) -> None:
    # Every call passes the parameter one same object, so the parameter is linked with it (these bodies were a
    # documented miss while named lambdas' parameters had no link). A value parameter, or another object, is not c.
    text = program(body)
    assert_flow(read({"main.cpp": text}), text)
    assert_clear(read({"main.cpp": program(body.replace(*copy))}))


FETCH_LAMBDA = (
    "    std::vector<float> staged(af, af + n);\n"
    "    auto fetch = [&](std::vector<float>& v) {\n"
    "        distributed::EnqueueReadMeshBuffer(cq, v, c_buf, true);\n"
    "    };\n"
)
FETCH_HELPER = (
    "static void fetch(distributed::MeshCommandQueue& cq, std::vector<float>& v,\n"
    "                  const std::shared_ptr<distributed::MeshBuffer>& buf) {\n"
    "    distributed::EnqueueReadMeshBuffer(cq, v, buf, true);\n"
    "}\n"
)


def test_a_device_read_through_a_named_lambdas_reference_parameter_clears_its_argument() -> None:
    # Found by the P4.12 commit audit, with its controls: an own free function with a reference parameter, and no
    # staging.
    assert_clear(read({"main.cpp": program(FETCH_LAMBDA + "    fetch(staged);\n    c = staged;\n")}))
    twice = FETCH_LAMBDA + "    fetch(staged);\n    fetch(staged);\n    c = staged;\n"
    assert_clear(read({"main.cpp": program(twice)}))
    staged = "    std::vector<float> staged(af, af + n);\n    fetch(cq, staged, c_buf);\n    c = staged;\n"
    assert_clear(read({"main.cpp": program(staged, helpers=FETCH_HELPER)}))
    unstaged = FETCH_LAMBDA.replace("staged(af, af + n)", "staged(n)") + "    fetch(staged);\n    c = staged;\n"
    assert_clear(read({"main.cpp": program(unstaged)}))
    # Refilling another vector leaves the staged input in place.
    text = program(FETCH_LAMBDA + "    fetch(res);\n    c = staged;\n")
    assert_flow(read({"main.cpp": text}), text)


STAGE_BUF = (
    "    std::vector<float> buf(af, af + n);\n"
    "    distributed::EnqueueWriteMeshBuffer(cq, in_buf, buf, false);\n"
)


def finish_with(call: str) -> str:
    """Return a named lambda finish that reads the device result back into buf and then runs `call`, and its call."""
    return (
        "    auto finish = [&]() {\n"
        "        distributed::EnqueueReadMeshBuffer(cq, buf, c_buf, true);\n"
        f"        {call}\n"
        "    };\n"
        "    finish();\n"
    )


@pytest.mark.parametrize(
    ("helper", "call"),
    [
        (
            "    auto copy_out = [](const std::vector<float>& src, std::vector<float>& dst) {\n"
            "        for (size_t i = 0; i < src.size(); ++i) {\n"
            "            dst[i] = src[i];\n"
            "        }\n"
            "    };\n",
            "copy_out(buf, c);",
        ),
        (
            "    auto unpack = [](const float* src, float* dst, uint32_t count) {\n"
            "        for (uint32_t i = 0; i < count; ++i) {\n"
            "            dst[i] = src[i];\n"
            "        }\n"
            "    };\n",
            "unpack(buf.data(), c.data(), n);",
        ),
        (
            "    auto copy_out = [&src = buf, &dst = c]() {\n"
            "        for (size_t i = 0; i < src.size(); ++i) {\n"
            "            dst[i] = src[i];\n"
            "        }\n"
            "    };\n",
            "copy_out();",
        ),
    ],
    ids=["reference-parameters", "pointer-parameters", "reference-init-captures"],
)
def test_a_named_lambda_called_inside_another_named_lambda_links_nothing(helper: str, call: str) -> None:
    # Found by the P4.12 commit audit: the conversion lambda, written before the read-back, is called only inside
    # finish, so it is read where it is written. Its parameters and reference init captures get no alias link
    # there, so its body, which sits before the device read, does not read the staged input through them.
    assert_clear(read({"main.cpp": program(STAGE_BUF + helper + finish_with(call))}))


# ---------------------------------------------------------------------------
# Other sources and sinks (_cxx_flow's module docstring, Events)


@pytest.mark.parametrize("callee", ["fwrite", "std::fwrite"])
def test_fwrite_of_host_computed_data_is_a_sink(callee: str) -> None:
    tail = (
        '    std::FILE* out = std::fopen(argv[3], "wb");\n'
        f"    {callee}(c.data(), sizeof(float), c.size(), out);\n"
        "    std::fclose(out);\n"
    )
    text = program(HOST_TO_C, tail=tail)
    assert_flow(read({"main.cpp": text}), text, sink=f"{callee}(")


@pytest.mark.parametrize("stream", ["std::ofstream", "std::fstream"])
def test_an_output_stream_write_is_a_sink(stream: str) -> None:
    tail = (
        f"    {stream} out(argv[3], std::ios::binary);\n"
        "    out.write(reinterpret_cast<const char*>(c.data()), static_cast<std::streamsize>(4 * n));\n"
    )
    text = program(HOST_TO_C, tail=tail)
    assert_flow(read({"main.cpp": text}), text, sink="out.write(")


def test_a_write_method_of_another_type_is_not_a_sink() -> None:
    tail = "    Logger out;\n    out.write(reinterpret_cast<const char*>(c.data()), 4 * n);\n"
    result = read({"main.cpp": program(HOST_TO_C, tail=tail)})
    assert result.host_compute is None and [summary(item) for item in result.diagnostics] == [no_output_summary()]


@pytest.mark.parametrize("callee", ["fread", "std::fread"])
def test_fread_input_copied_to_the_output_is_host_compute(callee: str) -> None:
    body = (
        '    std::FILE* in = std::fopen("extra.bin", "rb");\n'
        "    std::vector<float> extra(n);\n"
        f"    {callee}(extra.data(), sizeof(float), n, in);\n"
        "    std::fclose(in);\n"
        + READ_BACK
        + "    c = extra;\n"
    )
    text = program(body)
    assert_flow(read({"main.cpp": text}), text, source=f"{callee}(")


def test_an_input_stream_read_is_a_source() -> None:
    body = (
        '    std::ifstream in("extra.bin", std::ios::binary);\n'
        "    std::vector<float> extra(n);\n"
        "    in.read(reinterpret_cast<char*>(extra.data()), static_cast<std::streamsize>(4 * n));\n"
        + READ_BACK
        + "    c = extra;\n"
    )
    text = program(body)
    assert_flow(read({"main.cpp": text}), text, source="in.read(")


def test_posix_read_and_write_are_a_source_and_a_sink() -> None:
    body = (
        '    int in_fd = open("extra.bin", 0);\n'
        "    std::vector<float> extra(n);\n"
        "    read(in_fd, extra.data(), 4 * n);\n"
        + READ_BACK
        + "    c = extra;\n"
    )
    tail = "    int out_fd = open(argv[3], 1);\n    write(out_fd, c.data(), 4 * n);\n"
    text = program(body, tail=tail)
    assert_flow(read({"main.cpp": text}), text, source="read(in_fd", sink="write(out_fd")


def test_an_object_like_macro_alias_of_lassi_io_write_is_a_sink() -> None:
    tail = "    uint64_t dims[1] = {n};\n    WRITE(argv[3], \"c\", LASSI_IO_F32, 1u, dims, c.data());\n"
    text = program(HOST_TO_C, helpers="#define WRITE lassi_io_write\n", tail=tail)
    assert_flow(read({"main.cpp": text}), text, sink="WRITE(argv[3]")


def test_macro_aliases_expand_one_level_and_cycles_end() -> None:
    helpers = "#define FIRST SECOND\n#define SECOND FIRST\n"
    tail = "    uint64_t dims[1] = {n};\n    FIRST(argv[3], \"c\", LASSI_IO_F32, 1u, dims, c.data());\n"
    result = read({"main.cpp": program(HOST_TO_C, helpers=helpers, tail=tail)})
    assert result.host_compute is None and [summary(item) for item in result.diagnostics] == [no_output_summary()]


def test_lassi_io_write_with_five_arguments_is_not_a_sink() -> None:
    tail = "    uint64_t dims[1] = {n};\n    lassi_io_write(argv[3], LASSI_IO_F32, 1u, dims, c.data());\n"
    result = read({"main.cpp": program(HOST_TO_C, tail=tail)})
    assert result.host_compute is None and [summary(item) for item in result.diagnostics] == [no_output_summary()]


# ---------------------------------------------------------------------------
# The include closure (ttmetal_guard's Host text; _cxx_scan's Includes)


def test_a_quoted_include_looks_beside_the_including_file_first() -> None:
    text = program(COMBINE_BODY, includes='#include "kernels/ops.h"\n')
    files = {"host/main.cpp": text, "host/kernels/ops.h": VIOLATING_OPS, "kernels/ops.h": CLEAN_OPS}
    assert_flow(read(files), text, file="host/main.cpp")
    swapped = {**files, "host/kernels/ops.h": CLEAN_OPS, "kernels/ops.h": VIOLATING_OPS}
    assert_clear(read(swapped))


def test_an_angle_include_looks_only_at_the_build_directory_root() -> None:
    text = program(COMBINE_BODY, includes="#include <kernels/ops.h>\n")
    files = {"host/main.cpp": text, "host/kernels/ops.h": CLEAN_OPS, "kernels/ops.h": VIOLATING_OPS}
    assert_flow(read(files), text, file="host/main.cpp")
    swapped = {**files, "host/kernels/ops.h": VIOLATING_OPS, "kernels/ops.h": CLEAN_OPS}
    assert_clear(read(swapped))


def test_model_files_are_checked_before_harness_files() -> None:
    text = program(COMBINE_BODY, includes='#include "kernels/ops.h"\n')
    assert_clear(read({"main.cpp": text, "kernels/ops.h": CLEAN_OPS}, {"kernels/ops.h": VIOLATING_OPS}))
    assert_flow(read({"main.cpp": text}, {"kernels/ops.h": VIOLATING_OPS}), text)


def test_only_harness_files_that_host_code_includes_are_read() -> None:
    included = program(COMBINE_BODY, includes='#include "ops.h"\n')
    assert_flow(read({"main.cpp": included}, {"ops.h": VIOLATING_OPS}), included)
    assert_clear(read({"main.cpp": program(COMBINE_BODY)}, {"ops.h": VIOLATING_OPS}))


def test_an_include_inside_if_0_is_not_followed() -> None:
    text = program(COMBINE_BODY, includes='#if 0\n#include "kernels/ops.h"\n#endif\n')
    assert_clear(read({"main.cpp": text, "kernels/ops.h": VIOLATING_OPS}))


def test_include_cycles_end() -> None:
    first = '#include "b.h"\n' + VIOLATING_OPS
    second = SYNTHETIC_LINE + '\n#include "a.h"\n'
    text = program(COMBINE_BODY, includes='#include "kernels/a.h"\n')
    assert_flow(read({"main.cpp": text, "kernels/a.h": first, "kernels/b.h": second}), text)


def test_an_include_that_leaves_the_build_directory_is_not_read() -> None:
    text = program(COMBINE_BODY, includes='#include "../ops.h"\n')
    assert_clear(read({"main.cpp": text}, {"ops.h": VIOLATING_OPS}))


# ---------------------------------------------------------------------------
# Lexing seen through the reading (_cxx_scan's module docstring, Lexing)


def test_an_if_0_region_drops_the_sink_and_its_else_branch_keeps_it() -> None:
    dropped = program(HOST_TO_C, tail="#if 0\n" + WRITE_C + "#endif\n")
    result = read({"main.cpp": dropped})
    assert result.host_compute is None and [summary(item) for item in result.diagnostics] == [no_output_summary()]
    kept = program(HOST_TO_C, tail="#if false\n    int unused = 0;\n#else\n" + WRITE_C + "#endif\n")
    assert_flow(read({"main.cpp": kept}), kept)


def test_nested_conditionals_inside_if_0_are_dropped_with_it() -> None:
    tail = "#if 0\n#ifdef LASSI_EXTRA\n    int unused = 0;\n#endif\n" + WRITE_C + "#endif\n"
    result = read({"main.cpp": program(HOST_TO_C, tail=tail)})
    assert result.host_compute is None and [summary(item) for item in result.diagnostics] == [no_output_summary()]


def test_line_splices_keep_the_original_line_numbers() -> None:
    body = "    uint32_t spliced = 1u + \\\n        2u;\n    (void)spliced;\n    // a comment \\\n    that continues\n"
    text = program(body + HOST_TO_C, helpers="#define LONG_VALUE 1 + \\\n    2\n")
    assert_flow(read({"main.cpp": text}), text)


@pytest.mark.parametrize("newline", ["\r\n", "\r"], ids=["crlf", "cr"])
def test_crlf_and_lone_cr_read_as_line_feeds(newline: str) -> None:
    files = fixture_files("host_loop_output")
    converted = {path: text.replace("\n", newline) for path, text in files.items()}
    assert read(converted) == read(files)


def test_raw_strings_and_digit_separators_lex_correctly() -> None:
    body = (
        '    const char* banner = R"tag(a ")" quote and lassi_io_write(x) inside)tag";\n'
        "    (void)banner;\n"
        "    const uint64_t big = 1'000'000;\n"
        "    const char quote = '\\'';\n"
        "    (void)big;\n"
        "    (void)quote;\n"
    )
    assert_clear(read({"main.cpp": program(body + CLEAN_BODY)}))
    hidden = '    const char* raw = R"d(x )" CreateKernel ComputeConfig )d";\n    (void)raw;\n'
    text = program(hidden + CLEAN_BODY, kernels="")
    assert_no_kernel(read({"main.cpp": text}), text)


# ---------------------------------------------------------------------------
# Robustness and bounds (ttmetal_guard's Host text and The reading; _cxx_scan's Bounds)


@pytest.mark.parametrize(
    ("ending", "reason"),
    [
        ("/* never closed\n", "an unterminated comment in main.cpp"),
        ('const char* broken = "never closed;\nint after = 0;\n', "an unterminated literal in main.cpp"),
        ("const char broken = 'x;\nint after = 0;\n", "an unterminated literal in main.cpp"),
        ('const char* broken = R"d(never closed\n', "an unterminated literal in main.cpp"),
    ],
    ids=["comment", "string", "char", "raw-string"],
)
def test_unterminated_comments_and_literals_make_the_text_unreadable(ending: str, reason: str) -> None:
    result = read({"main.cpp": program(HOST_TO_C) + ending})
    assert result.host_compute is None
    assert [summary(item) for item in result.diagnostics] == [unreadable_summary(reason)]


def test_an_unreadable_included_file_is_named() -> None:
    text = program(COMBINE_BODY, includes='#include "kernels/ops.h"\n')
    result = read({"main.cpp": text, "kernels/ops.h": VIOLATING_OPS + "/* never closed\n"})
    assert result.host_compute is None
    reason = "an unterminated comment in kernels/ops.h"
    assert [summary(item) for item in result.diagnostics] == [unreadable_summary(reason)]


def test_the_first_unreadable_file_in_read_order_is_named() -> None:
    # The roots are read in sorted path order and the files they include after them, breadth-first: the root m.cpp
    # is read before kernels/k.h, which only a.cpp includes, although kernels/k.h sorts first.
    files = {
        "a.cpp": SYNTHETIC_LINE + '\n#include "kernels/k.h"\nint main() {\n    return 0;\n}\n',
        "kernels/k.h": SYNTHETIC_LINE + "\n/* never closed\n",
        "m.cpp": SYNTHETIC_LINE + '\nconst char* s = "never closed;\n',
    }
    result = read(files)
    assert result.host_compute is None
    assert [summary(item) for item in result.diagnostics] == [unreadable_summary("an unterminated literal in m.cpp")]


def size_reason() -> str:
    """Return the unreadable reason of host text over the size bound."""
    return f"more than {SIZE_LIMIT} bytes of host text"


def test_host_text_over_the_size_bound_is_not_checked() -> None:
    text = program(HOST_TO_C) + "/*" + "x" * SIZE_LIMIT + "*/\n"
    result = read({"main.cpp": text})
    assert result.host_compute is None
    assert [summary(item) for item in result.diagnostics] == [unreadable_summary(size_reason())]


def test_the_size_bound_sums_every_file_read() -> None:
    half = "/*" + "x" * (SIZE_LIMIT // 2 + 16) + "*/\n"
    text = program(COMBINE_BODY, includes='#include "kernels/ops.h"\n') + half
    result = read({"main.cpp": text}, {"kernels/ops.h": VIOLATING_OPS + half})
    assert [summary(item) for item in result.diagnostics] == [unreadable_summary(size_reason())]


def test_the_size_bound_counts_utf8_bytes() -> None:
    padding = "/*" + "\u00e9" * (SIZE_LIMIT // 2 + 16) + "*/\n"
    assert len(padding) < SIZE_LIMIT < len(padding.encode("utf-8"))
    result = read({"main.cpp": program(HOST_TO_C) + padding})
    assert [summary(item) for item in result.diagnostics] == [unreadable_summary(size_reason())]


def test_files_that_are_never_read_do_not_count_toward_the_size_bound() -> None:
    big = "/*" + "x" * (SIZE_LIMIT + 16) + "*/\n"
    files = {**fixture_files("clean_offload"), "kernels/big.cpp": big}
    assert_clear(read(files, {**harness_header(), "big.h": big}))


@pytest.mark.slow
def test_host_text_of_exactly_the_size_bound_is_read() -> None:
    base = program(CLEAN_BODY)
    padding = SIZE_LIMIT - len(base.encode("utf-8")) - len("/**/\n")
    text = base + "/*" + "x" * padding + "*/\n"
    assert len(text.encode("utf-8")) == SIZE_LIMIT
    assert_clear(read({"main.cpp": text}))


def nested(depth: int) -> str:
    """Return a file-scope declaration whose initializer nests `depth` parentheses."""
    return "static const int deep = " + "(" * depth + "1" + ")" * depth + ";\n"


def test_brackets_nested_to_the_bound_are_read() -> None:
    assert_clear(read({"main.cpp": program(CLEAN_BODY, helpers=nested(DEPTH_LIMIT))}))


@pytest.mark.parametrize("depth", [DEPTH_LIMIT + 1, 10000])
def test_brackets_nested_past_the_bound_make_the_text_unreadable(depth: int) -> None:
    result = read({"main.cpp": program(HOST_TO_C, helpers=nested(depth))})
    assert result.host_compute is None
    reason = f"brackets nested deeper than {DEPTH_LIMIT} in main.cpp"
    assert [summary(item) for item in result.diagnostics] == [unreadable_summary(reason)]


@pytest.mark.parametrize(
    "files",
    [
        {},
        {"main.cpp": ""},
        {"main.cpp": "}}}}))))]]]]"},
        {"main.cpp": "(((({{{{[[[["},
        {"main.cpp": "int main( {\n"},
        {"main.cpp": "#if 0\n"},
        {"main.cpp": "#endif\n#else\n"},
        {"main.cpp": "\\\n"},
        {"main.cpp": "int gr\u00f6\u00dfe = 1;\nconst char* s = \"\u00e9t\u00e9\";\nint main() { return 0; }\n"},
        {"main.cpp": "<% int x; %> <: :> %:define A B\n"},
        {"notes.txt": "lassi_io_write(a, b, c, d, e, f);\n"},
    ],
    ids=["no-files", "empty", "closers", "openers", "open-main", "open-if", "stray-else", "splice", "non-ascii",
         "digraphs", "no-host-file"],
)
def test_odd_inputs_never_raise(files: dict[str, str]) -> None:
    assert read(files).host_compute is None, "none of these host texts writes a recognized output"


def test_the_reader_stops_at_its_step_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    module = guard()
    monkeypatch.setattr(module, "STEP_LIMIT", 1000)
    result = read(fixture_files("clean_offload"), harness_header())
    assert result.host_compute is None
    reason = "the reader exceeded its budget of 1000 steps"
    assert [summary(item) for item in result.diagnostics] == [unreadable_summary(reason)]


def global_ring(count: int, chain: int) -> str:
    """Return a program whose `count` file-scope floats are assigned in a ring inside a loop, under `chain` callers.

    main stores an input value in the last float and reads the first after
    the call chain, so the reading is True; the ring makes every float's
    summary flow reach every other float.
    """
    ring = " ".join(f"g{i} = g{(i + 1) % count};" for i in range(count))
    helpers = "".join(f"float g{i};\n" for i in range(count))
    helpers += f"void f0() {{\n    for (int r = 0; r < 2; ++r) {{\n        {ring}\n    }}\n}}\n"
    helpers += "".join(f"void f{i}() {{\n    f{i - 1}();\n}}\n" for i in range(1, chain + 1))
    body = f"    g{count - 1} = af[0];\n    f{chain}();\n    c[0] = g0;\n"
    return program(body, helpers=helpers)


def many_parameters(count: int, uses: int) -> str:
    """Return a program with one called function of `count` defaulted parameters assigning a free name `uses` times."""
    params = ", ".join(f"int p{i} = 0" for i in range(count))
    helpers = f"static void wide({params}) {{\n" + "    q = q;\n" * uses + "}\n"
    return program("    wide();\n" + CLEAN_BODY, helpers=helpers)


def test_wide_programs_stay_within_the_step_budget() -> None:
    # Two wide shapes: 200 file-scope floats assigned in a ring under a call chain, so every summary flow reaches
    # every port, and one function with 2000 defaulted parameters. Both are read within the step budget.
    ring = global_ring(200, 4)
    assert_flow(read({"main.cpp": ring}), ring, sink=WRITE_NEEDLE)
    assert_clear(read({"main.cpp": many_parameters(2000, 8000)}))


def test_an_else_if_chain_whose_headers_declare_names_is_read_within_the_step_budget() -> None:
    # Each header's scope runs to the end of the chain (its else branch included), so the end of the chain is
    # searched from every header; it is found once (the P4.12 commit audit: 1000 branches exceeded the budget).
    chain = "    if (int s0 = 0) c[0] = res[0];\n" + "".join(
        f"    else if (int s{k} = {k}) c[{k % 4}] = res[{k % 4}];\n" for k in range(1, 1000)
    )
    assert_clear(read({"main.cpp": program(READ_BACK + chain)}))


def test_no_host_text_reads_as_no_output_write() -> None:
    for files in ({}, {"main.cpp": ""}, {"kernels/only.cpp": program(HOST_TO_C)}):
        result = read(files)
        assert result.host_compute is None and [summary(item) for item in result.diagnostics] == [no_output_summary()]


SOUP_WORDS = (
    "int", "float", "auto", "const", "static", "struct", "class", "namespace", "template", "typename",
    "operator", "return", "for", "if", "else", "while", "do", "switch", "case", "this", "sizeof", "decltype",
    "new", "delete", "main", "x", "y", "c", "a", "res", "af", "std", "copy", "transform", "for_each", "memcpy",
    "fill", "swap", "push_back", "lassi_io_read", "lassi_io_write", "fwrite", "fread", "read", "write", "data",
    "count", "dims", "size", "CreateKernel", "ComputeConfig", "DataMovementConfig", "KernelDescriptor",
    "EnqueueReadMeshBuffer", "ReadShard", "(", ")", "{", "}", "[", "]", "<", ">", ">>", "<<", "::", "->", ".",
    ",", ";", ":", "=", "+=", "==", "*", "&", "&&", "+", "-", "/", "%", "!", "?", "~", "...", "#", "##", "@",
    "`", "$", "<%", "%>", '"text"', "'c'", 'u8"bytes"', "L'w'", 'R"d(raw )" still raw)d"', "1'000'000",
    "0x1p+3", "1.5e-3f", "42u", "/* comment */", "// line comment\n", "\\\n", "#define A B\n",
    "#define F(x) x\n", '#include "util.h"\n', '#include "kernels/k.h"\n', "#include <vector>\n", "#if 0\n",
    "#if false\n", "#ifdef X\n", "#else\n", "#elif 1\n", "#endif\n", "[&]", "[=]", "[x, &y]", "\u00e9",
    "\u53d8\u91cf", '"\u00e9t\u00e9"', "\u00fc",
)
SOUP_GAPS = (" ", " ", " ", "\n", "\r\n", "\t", "\r")
SOUP_SEED = 20261001
SOUP_CASES = 200


def soup(rng: random.Random, words: int) -> str:
    """Return `words` random SOUP_WORDS joined by random SOUP_GAPS: text that is rarely C++ but always lexable."""
    return "".join(rng.choice(SOUP_WORDS) + rng.choice(SOUP_GAPS) for _ in range(words))


def test_seeded_token_soup_never_raises_and_never_reaches_the_reader_failure() -> None:
    rng = random.Random(SOUP_SEED)
    for _ in range(SOUP_CASES):
        files = {
            "main.cpp": soup(rng, 200),
            "util.h": soup(rng, 80),
            "kernels/k.h": soup(rng, 40),
            "kernels/k.cpp": soup(rng, 40),
        }
        read(files, {"lassi_io.h": soup(rng, 40)})


def test_the_reading_does_not_depend_on_dict_order() -> None:
    files = {
        **fixture_files("host_loop_output"),
        "util.h": VIOLATING_OPS,
        "zz/extra.cpp": SYNTHETIC_LINE + '\n#include "../util.h"\nvoid extra() {}\n',
        "kernels/ops.h": CLEAN_OPS,
    }
    harness = {**harness_header(), "unused.h": CLEAN_OPS}
    expected = read(dict(sorted(files.items())), dict(sorted(harness.items())))
    rng = random.Random(7)
    for _ in range(5):
        keys, names = list(files), list(harness)
        rng.shuffle(keys)
        rng.shuffle(names)
        assert read({key: files[key] for key in keys}, {name: harness[name] for name in names}) == expected
    assert read(dict(reversed(files.items())), dict(reversed(harness.items()))) == expected


def test_messages_are_plain_ascii_with_a_non_ascii_file_name_escaped() -> None:
    files = fixture_files("host_loop_output")
    name = "src/m\u00e4in.cpp"
    text = files.pop("main.cpp")
    result = read({**files, name: text})
    (note,) = result.diagnostics
    assert note.file == name, "the Diagnostic keeps the path as recorded"
    escaped = "src/m\\xe4in.cpp"
    expected = flow_message(f"{escaped}:{line_of(text, READ_A)}", f"{escaped}:{line_of(text, 'lassi_io_write(')}")
    assert note.message == expected and note.message.isascii()
    broken = read({name: program(HOST_TO_C) + "/* never closed\n"})
    assert [summary(item) for item in broken.diagnostics] == [
        unreadable_summary(f"an unterminated comment in {escaped}")
    ]


def test_the_arguments_may_be_read_only_mappings_and_are_left_unchanged() -> None:
    files = fixture_files("host_loop_output")
    harness = harness_header()
    before = (dict(files), dict(harness))
    result = guard().read_host_compute(MappingProxyType(files), MappingProxyType(harness))
    check_contract(result)
    assert result == read(files, harness) and (files, harness) == before


class SyntheticReaderDefect(Exception):
    """A SYNTHETIC defect injected into the reader."""


def test_a_defect_in_the_reader_reads_as_not_checked_naming_the_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    module = guard()

    def broken(path: str) -> bool:
        raise SyntheticReaderDefect(path)

    monkeypatch.setattr(module, "is_kernel_source", broken)
    result = read(fixture_files("host_loop_output"), defect=True)
    assert result.host_compute is None
    reason = "the reader failed with SyntheticReaderDefect"
    assert [summary(item) for item in result.diagnostics] == [unreadable_summary(reason)]


def test_an_interrupt_is_not_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    module = guard()

    def interrupted(path: str) -> bool:
        raise KeyboardInterrupt

    monkeypatch.setattr(module, "is_kernel_source", interrupted)
    with pytest.raises(KeyboardInterrupt):
        module.read_host_compute(fixture_files("host_loop_output"), {})


def refuse(name: str) -> Callable[..., Any]:
    """Return a stand-in for `name` that fails the test when the guard calls it."""

    def refused(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(f"the guard called {name}; it reads only its arguments and starts no process")

    return refused


def test_the_reader_opens_no_file_and_starts_no_process(monkeypatch: pytest.MonkeyPatch) -> None:
    files, harness = fixture_files("host_loop_output"), harness_header()
    expected = read(files, harness)
    for target, name in [
        (builtins, "open"),
        (io, "open"),
        (os, "open"),
        (os, "system"),
        (os, "popen"),
        (subprocess, "Popen"),
        (subprocess, "run"),
        (pathlib.Path, "open"),
        (pathlib.Path, "read_bytes"),
        (pathlib.Path, "read_text"),
    ]:
        monkeypatch.setattr(target, name, refuse(f"{getattr(target, '__name__', target)}.{name}"))
    result = guard().read_host_compute(files, harness)
    monkeypatch.undo()
    assert result == expected


# ---------------------------------------------------------------------------
# The stated limits (ttmetal_guard's module docstring, Limits): documented-limit tests, so a change is visible


def test_limit_laundering_through_the_device_is_missed_but_tagged() -> None:
    body = (
        "    std::vector<float> host_sum(n);\n"
        "    for (uint32_t i = 0; i < n; ++i) {\n"
        "        host_sum[i] = af[i] + bf[i];\n"
        "    }\n"
        "    distributed::EnqueueWriteMeshBuffer(cq, in_buf, host_sum, false);\n"
        + CLEAN_BODY
    )
    text = program(body, kernels=KERNELS_DM)
    assert_clear(read({"main.cpp": text}), tagged_at=line_of(text, "CreateKernel("))


def test_limit_a_dead_device_read_between_the_host_write_and_the_sink_clears_it() -> None:
    dead = "    if (false) {\n        distributed::EnqueueReadMeshBuffer(cq, c, c_buf, true);\n    }\n"
    assert_clear(read({"main.cpp": program(HOST_TO_C + dead)}))


def test_limit_a_compute_config_in_dead_code_clears_the_tag() -> None:
    dead = "    if (false) {\n        ComputeConfig unused{};\n        (void)unused;\n    }\n"
    assert_clear(read({"main.cpp": program(dead + CLEAN_BODY, kernels=KERNELS_DM)}))


def test_limit_a_kernel_creator_in_dead_code_counts() -> None:
    # With no kernel created, rule K would read True; the dead creator keeps it from firing.
    dead = "    if (false) {\n        KernelDescriptor unused{};\n        (void)unused;\n    }\n"
    text = program(dead + CLEAN_BODY, kernels="")
    assert_clear(read({"main.cpp": text}), tagged_at=line_of(text, "KernelDescriptor"))


def test_limit_a_kernel_created_but_never_launched_counts() -> None:
    # The inputs only pass through a device buffer; the compute kernel is created and never enqueued. The guard never
    # reads what a kernel computes (a kernel source is read only when host code includes it, and then as host text),
    # so this also pins a kernel that does no work: a created kernel counts whatever it does.
    kernels = (
        "    Program program = CreateProgram();\n"
        "    const CoreCoord core = {0, 0};\n"
        '    KernelHandle compute = CreateKernel(program, "kernels/compute.cpp", core, ComputeConfig{});\n'
        "    (void)compute;\n"
    )
    staged = (
        "    std::vector<float> staged(af, af + n);\n"
        "    distributed::EnqueueWriteMeshBuffer(cq, c_buf, staged, false);\n"
    )
    assert_clear(read({"main.cpp": program(staged + CLEAN_BODY, kernels=kernels)}))


@pytest.mark.parametrize(
    "body",
    [
        READ_BACK + "    for (uint32_t i = 0; i < n; ++i) {\n        c[i] = res[static_cast<int>(af[i]) % n];\n    }\n",
        CLEAN_BODY + "    if (af[0] > 0.0f) {\n        c[0] = 1.0f;\n    }\n",
        CLEAN_BODY + "    for (uint32_t i = 0; i < static_cast<uint32_t>(bf[0]); ++i) {\n        c[i] = 2.0f;\n    }\n",
        CLEAN_BODY + "    c.resize(static_cast<uint32_t>(bf[0]));\n",
        CLEAN_BODY + "    c[0] = static_cast<float>(a.dims[0] + b.count);\n",
    ],
    ids=["index", "branch", "loop-bound", "count", "metadata"],
)
def test_limit_index_control_count_and_metadata_flows_are_missed(body: str) -> None:
    assert_clear(read({"main.cpp": program(body)}))


@pytest.mark.parametrize(
    ("helpers", "call"),
    [
        (
            '#define EMIT(path, data) lassi_io_write(path, "c", LASSI_IO_F32, 1u, dims, data)\n',
            "EMIT(argv[3], c.data())",
        ),
        ("#define CAT(x, y) x##y\n", 'CAT(lassi_io_, write)(argv[3], "c", LASSI_IO_F32, 1u, dims, c.data())'),
    ],
    ids=["function-like", "token-pasting"],
)
def test_limit_a_function_like_macro_hides_its_sink(helpers: str, call: str) -> None:
    tail = f"    uint64_t dims[1] = {{n}};\n    {call};\n"
    result = read({"main.cpp": program(HOST_TO_C, helpers=helpers, tail=tail)})
    assert result.host_compute is None and [summary(item) for item in result.diagnostics] == [no_output_summary()]


def test_limit_a_call_through_a_function_pointer_is_unresolved() -> None:
    body = "    auto op = &add;\n    op(af, bf, c.data(), n);\n"
    assert_clear(read({"main.cpp": program(body, helpers=ADD_HELPER)}))


ADD_ONE_FUNCTOR = """\
struct AddOne {
    void operator()(const float* x, float* z, uint32_t count) const {
        for (uint32_t i = 0; i < count; ++i) {
            z[i] = x[i] + 1.0f;
        }
    }
};
"""


@pytest.mark.parametrize(
    "call",
    [
        "    AddOne add_one;\n    add_one(af, c.data(), n);\n",
        "    AddOne{}(af, c.data(), n);\n",
        "    AddOne add_one;\n    add_one.operator()(af, c.data(), n);\n",
    ],
    ids=["named-object", "temporary", "explicit-operator"],
)
def test_limit_a_functor_call_is_missed(call: str) -> None:
    # A call of an object's operator() is matched by the object's name or by nothing, never by `operator()`.
    assert_clear(read({"main.cpp": program(call, helpers=ADD_ONE_FUNCTOR)}))


@pytest.mark.parametrize(
    "body",
    [
        "    auto first = [&]() {\n        return af[0];\n    };\n    c[0] = first();\n",
        "    auto first = [x = af[0]]() {\n        return x;\n    };\n    c[0] = first();\n",
        "    [&c, x = af[0]]() {\n        c[0] = x;\n    }();\n",
    ],
    ids=["reference-capture", "init-capture-returned", "init-capture-written"],
)
def test_limit_a_value_a_lambda_returns_from_its_captures_or_holds_in_a_value_init_capture_is_missed(
    body: str,
) -> None:
    # A lambda's return defines no RET, and a call of it contributes only its arguments, which here are none; a value
    # init capture's name is declared in the lambda with no value. (The reference init capture case, `[&r = c]`, is
    # linked with c now: test_a_write_through_a_reference_binding_or_init_capture_reaches_the_bound_object.)
    assert_clear(read({"main.cpp": program(READ_BACK + DEVICE_TO_C + body)}))


def test_limit_methods_of_a_class_defined_in_a_function_body_are_never_read() -> None:
    body = (
        "    struct Local {\n"
        "        static void add_one(const float* x, float* z, uint32_t count) {\n"
        "            for (uint32_t i = 0; i < count; ++i) {\n"
        "                z[i] = x[i] + 1.0f;\n"
        "            }\n"
        "        }\n"
        "    };\n"
        "    Local::add_one(af, c.data(), n);\n"
    )
    assert_clear(read({"main.cpp": program(body)}))


def test_limit_a_pointer_re_pointed_from_a_device_buffer_to_an_input_is_cleared_by_the_device_read() -> None:
    # The re-point joins res and the input in one alias class for the whole function, so the device read into res,
    # which comes before it in the text, kills the input too.
    body = (
        READ_BACK
        + "    const float* p = res.data();\n"
        "    p = af;\n"
        "    for (uint32_t i = 0; i < n; ++i) {\n"
        "        c[i] = p[i];\n"
        "    }\n"
    )
    assert_clear(read({"main.cpp": program(body)}))


def test_limit_an_unknown_library_call_writing_through_a_pointer_is_missed() -> None:
    assert_clear(read({"main.cpp": program("    external_add(af, bf, c.data(), n);\n")}))
    # A std::back_inserter copy target is a call, so it has no base and the copy defines nothing.
    inserter = "    std::vector<float> out;\n    std::copy(af, af + n, std::back_inserter(out));\n"
    assert_clear(read({"main.cpp": program(inserter, tail=WRITE_OUT)}))


def test_limit_a_reference_bound_to_a_call_has_no_base() -> None:
    body = "    auto&& r = std::move(c);\n    for (uint32_t i = 0; i < n; ++i) {\n        r[i] = af[i];\n    }\n"
    assert_clear(read({"main.cpp": program(body)}))


# Own helpers that return a pointer into, or a reference to, their argument, and one that returns new values.
FIRST_OF = "static float* first_of(std::vector<float>& v) {\n    return v.data();\n}\n"
PICK = "static std::vector<float>& pick(std::vector<float>& v) {\n    return v;\n}\n"
TWICE = """\
static std::vector<float> twice(const float* x, uint32_t count) {
    std::vector<float> v(count);
    for (uint32_t i = 0; i < count; ++i) {
        v[i] = 2.0f * x[i];
    }
    return v;
}
"""
CALL_HELPERS = FIRST_OF + PICK + TWICE


@pytest.mark.parametrize(
    ("body", "direct"),
    [
        (
            "    float* p = first_of(c);\n    for (uint32_t i = 0; i < n; ++i) {\n        p[i] = af[i];\n    }\n",
            ("first_of(c)", "c.data()"),
        ),
        ("    for (float& v : pick(c)) {\n        v = af[0];\n    }\n", ("pick(c)", "c")),
        ("    const auto& r = twice(af, n);\n    c = r;\n", ("const auto& r", "const auto r")),
        (
            "    uint32_t k = 0;\n    for (const auto& v : twice(af, n)) {\n        c[k++] = v;\n    }\n",
            ("const auto& v", "auto v"),
        ),
        (
            "    std::vector<float> spare(n);\n"
            "    auto& dst = n > 0 ? c : spare;\n"
            "    for (uint32_t i = 0; i < n; ++i) {\n"
            "        dst[i] = af[i];\n"
            "    }\n",
            ("n > 0 ? c : spare", "c"),
        ),
        (
            "    const float* p = n > 0 ? af : bf;\n"
            "    for (uint32_t i = 0; i < n; ++i) {\n"
            "        c[i] = p[i];\n"
            "    }\n",
            ("n > 0 ? af : bf", "af"),
        ),
        (
            "    std::pair<std::vector<float>, int> p{std::vector<float>(n), 0};\n"
            "    std::pair<std::vector<float>, int> q{std::vector<float>(n), 0};\n"
            "    auto& [values, status] = n > 0 ? p : q;\n"
            "    (void)status;\n"
            "    for (uint32_t i = 0; i < n; ++i) {\n"
            "        values[i] = af[i];\n"
            "    }\n"
            "    c = p.first;\n",
            ("n > 0 ? p : q", "p"),
        ),
    ],
    ids=[
        "pointer-write",
        "range-for-element-write",
        "reference-read",
        "range-for-element-read",
        "conditional-write",
        "conditional-read",
        "binding-of-a-conditional-write",
    ],
)
def test_limit_a_pointer_or_reference_bound_to_a_call_or_a_conditional_is_missed(
    body: str, direct: tuple[str, str]
) -> None:
    # A call has no base, so a pointer, a reference, or a range-for reference element bound to one is linked with
    # nothing and holds no value; a conditional's base is its first mention, here its condition's n, for a reference
    # structured binding too. Binding the object itself, or by value, reads True.
    assert_clear(read({"main.cpp": program(body, helpers=CALL_HELPERS)}))
    text = program(body.replace(*direct), helpers=CALL_HELPERS)
    assert_flow(read({"main.cpp": text}), text)


COPY_INTO = """\
static void copy_into(std::span<const float> src, std::span<float> dst) {
    for (size_t i = 0; i < src.size(); ++i) {
        dst[i] = src[i];
    }
}
"""
# Own helpers that write their pointer or reference parameter from the input.
PUT_POINTER = """\
static void put(float* y, const float* x, uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
        y[i] = x[i];
    }
}
"""
PUT_REFERENCE = """\
static void put_all(std::vector<float>& y, const float* x) {
    for (uint32_t i = 0; i < y.size(); ++i) {
        y[i] = x[i];
    }
}
"""
COPY_INTO_CALL = "    copy_into(std::span<const float>(af, n), {argument});\n"


@pytest.mark.parametrize(
    ("call", "argument", "direct", "helpers"),
    [
        (COPY_INTO_CALL, "std::span<float>(c)", "c", COPY_INTO),
        (COPY_INTO_CALL, "std::span<float>{c}", "c", COPY_INTO),
        (COPY_INTO_CALL, "pick(c)", "c", COPY_INTO + PICK),
        ("    put({argument}, af, n);\n", "first_of(c)", "c.data()", PUT_POINTER + FIRST_OF),
        ("    put_all({argument}, af);\n", "pick(c)", "c", PUT_REFERENCE + PICK),
    ],
    ids=["view-temporary", "braced-view-temporary", "call", "pointer-parameter", "reference-parameter"],
)
def test_limit_a_write_through_an_own_function_parameter_whose_argument_has_no_base_is_missed(
    call: str, argument: str, direct: str, helpers: str
) -> None:
    # An out-port binds to base(argument) at the call; a view temporary and a call have no base, so the write through
    # a view, pointer, or reference parameter reaches no object of the caller. Passing the vector itself (or its
    # data()) reads True.
    assert_clear(read({"main.cpp": program(call.format(argument=argument), helpers=helpers)}))
    text = program(call.format(argument=direct), helpers=helpers)
    assert_flow(read({"main.cpp": text}), text)


def test_limit_a_models_own_lassi_io_wrapper_is_never_read() -> None:
    helpers = (
        "static int lassi_io_save(const char* path, const std::vector<float>& v) {\n"
        "    uint64_t shape[1] = {v.size()};\n"
        '    return lassi_io_write(path, "c", LASSI_IO_F32, 1u, shape, v.data());\n'
        "}\n"
    )
    result = read({"main.cpp": program(HOST_TO_C, helpers=helpers, tail="    lassi_io_save(argv[3], c);\n")})
    assert result.host_compute is None and [summary(item) for item in result.diagnostics] == [no_output_summary()]


@pytest.mark.parametrize(
    "tail",
    [
        '    std::FILE* out = std::fopen(argv[3], "w");\n    std::fprintf(out, "%f", c[0]);\n    std::fclose(out);\n',
        "    std::ofstream out(argv[3]);\n    out << c[0];\n",
        "    std::FILE* out = std::fopen(argv[3], \"wb\");\n    std::fputc(static_cast<int>(c[0]), out);\n",
        '    std::filesystem::copy_file("c.bin", argv[3]);\n',
        '    std::system("cp c.bin out.bin");\n',
        "    int fd = open(argv[3], 2);\n"
        "    float* out = static_cast<float*>(mmap(nullptr, 4 * n, 3, 1, fd, 0));\n"
        "    for (uint32_t i = 0; i < n; ++i) {\n        out[i] = c[i];\n    }\n",
    ],
    ids=["fprintf", "stream-insert", "fputc", "filesystem-copy", "system", "mmap"],
)
def test_limit_output_channels_outside_the_sink_list_read_as_not_checked(tail: str) -> None:
    result = read({"main.cpp": program(HOST_TO_C, tail=tail)})
    assert result.host_compute is None and [summary(item) for item in result.diagnostics] == [no_output_summary()]


def test_limit_input_channels_outside_the_source_list_give_no_flow() -> None:
    body = (
        '    std::ifstream in("extra.txt");\n'
        "    float first = 0.0f;\n"
        "    in >> first;\n"
        "    for (uint32_t i = 0; i < n; ++i) {\n"
        "        c[i] = first;\n"
        "    }\n"
    )
    assert_clear(read({"main.cpp": program(body)}))
    line = '    std::ifstream text("extra.txt");\n    std::string word;\n    std::getline(text, word);\n'
    assert_clear(read({"main.cpp": program(line + "    c[0] = static_cast<float>(word[0]);\n")}))
    mapped = (
        '    int fd = open("extra.bin", 0);\n'
        "    const float* m = static_cast<const float*>(mmap(nullptr, 4 * n, 1, 1, fd, 0));\n"
        "    for (uint32_t i = 0; i < n; ++i) {\n"
        "        c[i] = m[i];\n"
        "    }\n"
    )
    assert_clear(read({"main.cpp": program(mapped)}))


STREAM_DATA = "reinterpret_cast<const char*>(c.data())"
SAVER = """\
struct Saver {
    std::ofstream file;
    void save(const float* p, uint32_t count) {
        file.write(reinterpret_cast<const char*>(p), 4 * count);
    }
};
"""


@pytest.mark.parametrize(
    ("helpers", "tail"),
    [
        ("", f"    auto out = std::ofstream(argv[3], std::ios::binary);\n    out.write({STREAM_DATA}, 4 * n);\n"),
        ("", f"    std::ofstream(argv[3], std::ios::binary).write({STREAM_DATA}, 4 * n);\n"),
        ("", f"    std::basic_ofstream<char> out(argv[3]);\n    out.write({STREAM_DATA}, 4 * n);\n"),
        (SAVER, "    Saver s;\n    s.file.open(argv[3]);\n    s.save(c.data(), n);\n"),
        (SAVER, f"    Saver s;\n    s.file.open(argv[3]);\n    s.file.write({STREAM_DATA}, 4 * n);\n"),
    ],
    ids=["auto", "temporary", "basic-ofstream", "data-member-in-a-method", "data-member-of-an-object"],
)
def test_limit_a_stream_write_on_an_object_not_declared_as_a_listed_stream_is_no_sink(helpers: str, tail: str) -> None:
    # Only base(X)'s own declared type names are read: auto, a temporary, a data member (THIS or the object it is
    # a member of), and basic_ofstream name no ofstream, fstream, or ostream.
    result = read({"main.cpp": program(HOST_TO_C, helpers=helpers, tail=tail)})
    assert result.host_compute is None and [summary(item) for item in result.diagnostics] == [no_output_summary()]


def test_limit_a_stream_read_on_an_object_not_declared_as_a_listed_stream_is_no_source() -> None:
    body = (
        '    auto in = std::ifstream("extra.bin", std::ios::binary);\n'
        "    std::vector<float> extra(n);\n"
        "    in.read(reinterpret_cast<char*>(extra.data()), static_cast<std::streamsize>(4 * n));\n"
        + READ_BACK
        + "    c = extra;\n"
    )
    assert_clear(read({"main.cpp": program(body)}))


@pytest.mark.parametrize(
    ("helpers", "body", "tail"),
    [
        ("#define OUT c\n", HOST_TO_C.replace("c[i]", "OUT[i]"), WRITE_C),
        (
            '#define WRITE_C_OUT lassi_io_write(argv[3], "c", LASSI_IO_F32, 1u, dims, c.data())\n',
            HOST_TO_C,
            "    uint64_t dims[1] = {n};\n    WRITE_C_OUT;\n",
        ),
        ("#define PUT write\n", HOST_TO_C, f"    std::ofstream out(argv[3]);\n    out.PUT({STREAM_DATA}, 4 * n);\n"),
    ],
    ids=["variable", "sink-in-the-body", "member-callee"],
)
def test_limit_an_object_like_macro_hides_a_variable_or_a_sink(helpers: str, body: str, tail: str) -> None:
    # No object-like macro is expanded except a one-identifier alias in an unqualified free callee's name.
    result = read({"main.cpp": program(body, helpers=helpers, tail=tail)})
    if tail == WRITE_C:
        assert_clear(result)
    else:
        assert result.host_compute is None and [summary(item) for item in result.diagnostics] == [no_output_summary()]


def test_limit_a_loop_made_with_goto_has_no_back_edge() -> None:
    # The input reaches c only around the goto's back edge; the same body as a for loop reads True.
    body = (
        CLEAN_BODY
        + "    float staged = 0.0f;\n"
        "    int round = 0;\n"
        "again:\n"
        "    c[0] = staged;\n"
        "    staged = af[0];\n"
        "    if (++round < 2) {\n"
        "        goto again;\n"
        "    }\n"
    )
    assert_clear(read({"main.cpp": program(body)}))
    looped = program(
        body.replace("again:\n", "    for (; round < 2; ++round) {\n").replace(
            "    if (++round < 2) {\n        goto again;\n    }\n", "    }\n"
        )
    )
    assert_flow(read({"main.cpp": looped}), looped)


def test_limit_a_named_lambda_parameter_whose_calls_pass_different_objects_is_not_linked() -> None:
    # A pointer or reference parameter is linked with its argument only when every call passes one same base; here
    # the two calls pass different objects, so the parameter is defined from each argument and linked with neither.
    add = (
        "    std::vector<float> other(n);\n"
        "    auto add = [](const float* x, std::vector<float>& out) {\n"
        "        for (uint32_t i = 0; i < out.size(); ++i) {\n"
        "            out[i] = x[i] + 1.0f;\n"
        "        }\n"
        "    };\n"
        "    add(af, c);\n"
    )
    # The miss: a write through the parameter reaches neither argument (one call reads True), also for a
    # std::function variable initialized with the lambda.
    single = program(add)
    assert_flow(read({"main.cpp": single}), single)
    assert_clear(read({"main.cpp": program(add + "    add(bf, other);\n")}))
    function = add.replace("auto add = [](", "std::function<void(const float*, std::vector<float>&)> add = [](")
    single = program(function)
    assert_flow(read({"main.cpp": single}), single)
    assert_clear(read({"main.cpp": program(function + "    add(bf, other);\n")}))
    # The false positive: a device read through the parameter clears neither argument (one call reads False).
    twice = FETCH_LAMBDA + "    std::vector<float> other(n);\n    fetch(staged);\n    fetch(other);\n    c = staged;\n"
    text = program(twice)
    assert_flow(read({"main.cpp": text}), text)
    assert_clear(read({"main.cpp": program(twice.replace("fetch(other);", "fetch(staged);"))}))
    # An argument with no base (a call) gives no link even from a single call: the miss and the false positive.
    assert_clear(read({"main.cpp": program(add.replace("add(af, c);", "add(af, pick(c));"), helpers=PICK)}))
    text = program(FETCH_LAMBDA + "    fetch(pick(staged));\n    c = staged;\n", helpers=PICK)
    assert_flow(read({"main.cpp": text}), text)


def run_with(calls: str) -> str:
    """Return a named lambda run whose body is `calls` (statements, each on its own line), and its call."""
    inner = "".join("    " + line for line in calls.splitlines(keepends=True))
    return "    auto run = [&]() {\n" + inner + "    };\n    run();\n"


UNPACK_LAMBDA = (
    "    auto unpack = [](const float* src, float* dst, uint32_t count) {\n"
    "        for (uint32_t i = 0; i < count; ++i) {\n"
    "            dst[i] = src[i] + 1.0f;\n"
    "        }\n"
    "    };\n"
)
UNPACK_CALL = "    unpack(af, c.data(), n);\n"
ADD_INTO_LAMBDA = (
    "    auto add_into = [](std::vector<float>& out, const float* x, uint32_t count) {\n"
    "        for (uint32_t i = 0; i < count; ++i) {\n"
    "            out[i] = x[i] + 1.0f;\n"
    "        }\n"
    "    };\n"
)
FILL_CAPTURE_LAMBDA = (
    "    auto fill = [&, &dst = c]() {\n"
    "        for (uint32_t i = 0; i < n; ++i) {\n"
    "            dst[i] = af[i] + 1.0f;\n"
    "        }\n"
    "    };\n"
)
# A std::function lambda with a call inside its own body.
RECURSIVE_PUT = (
    "    std::function<void(const float*, float*, uint32_t)> put = [&](const float* x, float* z, uint32_t k) {\n"
    "        if (k == 0) {\n"
    "            return;\n"
    "        }\n"
    "        z[k - 1] = x[k - 1];\n"
    "        put(x, z, k - 1);\n"
    "    };\n"
    "    put(af, c.data(), n);\n"
)


@pytest.mark.parametrize(
    ("nested", "direct"),
    [
        (UNPACK_LAMBDA + run_with(UNPACK_CALL), UNPACK_LAMBDA + UNPACK_CALL),
        (ADD_INTO_LAMBDA + run_with("    add_into(c, af, n);\n"), ADD_INTO_LAMBDA + "    add_into(c, af, n);\n"),
        (FILL_CAPTURE_LAMBDA + run_with("    fill();\n"), FILL_CAPTURE_LAMBDA + "    fill();\n"),
        (run_with(UNPACK_LAMBDA + UNPACK_CALL), UNPACK_LAMBDA + UNPACK_CALL),
        (RECURSIVE_PUT, RECURSIVE_PUT.replace("        put(x, z, k - 1);\n", "")),
    ],
    ids=["pointer-parameter", "reference-parameter", "reference-init-capture", "written-inside", "calls-itself"],
)
def test_limit_a_write_through_a_named_lambda_called_inside_another_named_lambda_is_missed(
    nested: str, direct: str
) -> None:
    # The inner lambda is called inside run's body, is written inside it, or calls itself, so it is read where it is
    # written and its parameters and reference init captures are linked with nothing: the write reaches no argument.
    # The same lambda written and called outside any named lambda, with no call of itself, runs at its call, linked,
    # and reads True.
    assert_clear(read({"main.cpp": program(nested)}))
    text = program(direct)
    assert_flow(read({"main.cpp": text}), text)


IN_FROM_A = "    std::vector<float> in(af, af + n);\n"


@pytest.mark.parametrize(
    ("helper", "calls"),
    [
        (
            IN_FROM_A + "    auto convert = [&](const std::vector<float>& src) {\n"
            "        for (uint32_t i = 0; i < n; ++i) {\n"
            "            c[i] = src[i] * 2.0f;\n"
            "        }\n"
            "    };\n",
            "    convert(in);\n",
        ),
        ("    auto put = [&](float x) {\n        c[0] = x;\n    };\n", "    put(af[0]);\n"),
        (
            IN_FROM_A + "    auto copy_in = [&, &src = in]() {\n"
            "        for (uint32_t i = 0; i < n; ++i) {\n"
            "            c[i] = src[i];\n"
            "        }\n"
            "    };\n",
            "    copy_in();\n",
        ),
    ],
    ids=["reference-parameter", "value-parameter", "reference-init-capture"],
)
def test_limit_a_read_through_a_named_lambda_called_inside_another_named_lambda_is_missed(
    helper: str, calls: str
) -> None:
    # The inner lambda's body is read where it is written, before the call inside run that defines its parameters,
    # and its reference init capture holds no value, so the read through either gets none of the input. The same
    # call made outside any named lambda runs the lambda at the call and reads True.
    assert_clear(read({"main.cpp": program(helper + run_with(calls))}))
    text = program(helper + calls)
    assert_flow(read({"main.cpp": text}), text)


PUT_IN_LOOP = (
    "    for (uint32_t i = 0; i < n; ++i) {\n"
    "        auto put = [&](float x) {\n"
    "            c[i] = x;\n"
    "        };\n"
    "        put(af[i]);\n"
    "    }\n"
)
PUT_BEFORE_LOOP = (
    "    auto put = [&](uint32_t i, float x) {\n"
    "        c[i] = x;\n"
    "    };\n"
    "    for (uint32_t i = 0; i < n; ++i) {\n"
    "        put(i, af[i]);\n"
    "    }\n"
)
PUT_AND_GO_IN_LOOP = (
    "    for (uint32_t i = 0; i < n; ++i) {\n"
    "        auto put = [&](float x) {\n"
    "            c[i] = x;\n"
    "        };\n"
    "        auto go = [&]() {\n"
    "            put(af[i]);\n"
    "        };\n"
    "        go();\n"
    "    }\n"
)
PUT_AND_GO_BEFORE_LOOP = (
    "    auto put = [&](uint32_t i, float x) {\n"
    "        c[i] = x;\n"
    "    };\n"
    "    auto go = [&](uint32_t i) {\n"
    "        put(i, af[i]);\n"
    "    };\n"
    "    for (uint32_t i = 0; i < n; ++i) {\n"
    "        go(i);\n"
    "    }\n"
)
# A std::function lambda that reads its value parameter after the call inside its own body.
RECURSIVE_READ_AFTER = (
    "    std::function<void(uint32_t, float)> put = [&](uint32_t k, float x) {\n"
    "        if (k == 0) {\n"
    "            return;\n"
    "        }\n"
    "        put(k - 1, af[k - 1]);\n"
    "        c[k - 1] = x;\n"
    "    };\n"
    "    put(n, 0.0f);\n"
)
RECURSIVE_READ_BEFORE = RECURSIVE_READ_AFTER.replace(
    "        put(k - 1, af[k - 1]);\n        c[k - 1] = x;\n", "        c[k - 1] = x;\n        put(k - 1, af[k - 1]);\n"
)
# put is written and called inside go's body; go runs at its calls, and put's body is read with go's body there.
PUT_INSIDE_GO = (
    "    auto go = [&](uint32_t i) {\n"
    "        auto put = [&](float x) {\n"
    "            c[i] = x;\n"
    "        };\n"
    "        put(af[i]);\n"
    "    };\n"
)
GO_ONCE = PUT_INSIDE_GO + "    go(0);\n"


@pytest.mark.parametrize(
    ("reaches", "misses"),
    [
        (run_with(PUT_IN_LOOP), run_with(PUT_BEFORE_LOOP)),
        (PUT_AND_GO_IN_LOOP, PUT_AND_GO_BEFORE_LOOP),
        (RECURSIVE_READ_AFTER, RECURSIVE_READ_BEFORE),
        (PUT_INSIDE_GO + "    go(0);\n    go(1);\n", GO_ONCE),
        (PUT_INSIDE_GO + "    for (uint32_t i = 0; i < n; ++i) {\n        go(i);\n    }\n", GO_ONCE),
    ],
    ids=["loop-in-run", "called-inside-go-in-a-loop", "calls-itself", "inside-go-called-twice", "inside-go-in-a-loop"],
)
def test_limit_a_read_through_a_named_lambda_called_inside_another_gets_a_definition_that_reaches_it(
    reaches: str, misses: str
) -> None:
    # The inner lambda is read where it is written (or, written inside go's or run's body, with that body at each of
    # that lambda's calls), and each of its parameters is defined from a call's argument at that call. A loop holding
    # both the body and the call carries the definition around its back edge, a call inside the body (the lambda's
    # own) defines it for the reads after it, and a definition made in one run of go's body reaches the reads of a
    # later run (go's second call, or go's call again around a loop's back edge): each reads True. With only the call
    # in the loop, the read before the lambda's own call, or go called once outside any loop, the definition never
    # reaches the read: the miss reads False.
    text = program(reaches)
    assert_flow(read({"main.cpp": text}), text)
    assert_clear(read({"main.cpp": program(misses)}))


STAGED_FROM_A = "    std::vector<float> staged(af, af + n);\n"
FETCH_CALLS = "    fetch(staged);\n    c = staged;\n"
FETCH_POINTER_LAMBDA = (
    STAGED_FROM_A + "    auto fetch = [&](float* p) {\n"
    "        distributed::EnqueueReadMeshBuffer(cq, p, c_buf, true);\n"
    "    };\n"
)
FETCH_CAPTURE_LAMBDA = (
    STAGED_FROM_A + "    auto fetch = [&, &v = staged]() {\n"
    "        distributed::EnqueueReadMeshBuffer(cq, v, c_buf, true);\n"
    "    };\n"
)
FETCH_ONLY = FETCH_LAMBDA.replace(STAGED_FROM_A, "")


@pytest.mark.parametrize(
    ("nested", "direct"),
    [
        (FETCH_LAMBDA + run_with(FETCH_CALLS), FETCH_LAMBDA + FETCH_CALLS),
        (
            FETCH_POINTER_LAMBDA + run_with("    fetch(staged.data());\n    c = staged;\n"),
            FETCH_POINTER_LAMBDA + "    fetch(staged.data());\n    c = staged;\n",
        ),
        (
            FETCH_CAPTURE_LAMBDA + run_with("    fetch();\n    c = staged;\n"),
            FETCH_CAPTURE_LAMBDA + "    fetch();\n    c = staged;\n",
        ),
        (STAGED_FROM_A + run_with(FETCH_ONLY + FETCH_CALLS), FETCH_LAMBDA + FETCH_CALLS),
    ],
    ids=["reference-parameter", "pointer-parameter", "reference-init-capture", "written-inside"],
)
def test_limit_false_positive_a_device_read_through_a_named_lambda_called_inside_another_kills_only_that_name(
    nested: str, direct: str
) -> None:
    # fetch is called or written inside run's body, so its parameter or reference init capture is linked with nothing
    # and the device read through it leaves the staged input in staged; the same lambda written and called outside
    # any named lambda reads False.
    text = program(nested)
    assert_flow(read({"main.cpp": text}), text)
    assert_clear(read({"main.cpp": program(direct)}))


@pytest.mark.parametrize(
    "call",
    [
        "std::merge(af, af + n, bf, bf + n, c.begin());",
        "std::rotate_copy(af, af + 1, af + n, c.begin());",
        "std::uninitialized_copy(af, af + n, c.begin());",
        "std::set_union(af, af + n, bf, bf + n, c.begin());",
    ],
    ids=["merge", "rotate_copy", "uninitialized_copy", "set_union"],
)
def test_limit_a_standard_algorithm_outside_the_table_is_missed(call: str) -> None:
    # Only the standard-call table of _cxx_flow's module docstring defines a target; std::copy into c reads True.
    assert_clear(read({"main.cpp": program(f"    {call}\n")}))


def test_limit_ranges_for_each_is_missed() -> None:
    # Only std::for_each and for_each_n bind their lambda's parameter (the std::for_each form reads True).
    body = "    std::ranges::for_each(c, [&](float& v) {\n        v = af[0];\n    });\n"
    assert_clear(read({"main.cpp": program(body)}))


PUT_LAMBDA = "    auto put = [&](float& v) {\n        v = af[0];\n    };\n"


@pytest.mark.parametrize(
    "body",
    [
        "    std::invoke([&](std::vector<float>& v) {\n"
        "        for (uint32_t i = 0; i < n; ++i) {\n"
        "            v[i] = af[i];\n"
        "        }\n"
        "    }, c);\n",
        PUT_LAMBDA + "    std::for_each(c.begin(), c.end(), put);\n",
        PUT_LAMBDA + "    std::for_each_n(c.begin(), n, put);\n",
    ],
    ids=["invoke", "for-each-lambda-variable", "for-each-n-lambda-variable"],
)
def test_limit_invoke_and_a_for_each_given_a_lambda_variable_are_missed(body: str) -> None:
    # std::invoke binds no lambda parameter, and std::for_each and for_each_n bind one only when their last argument
    # is a lambda expression.
    assert_clear(read({"main.cpp": program(body)}))
    inline = program("    std::for_each(c.begin(), c.end(), [&](float& v) {\n        v = af[0];\n    });\n")
    assert_flow(read({"main.cpp": inline}), inline)


SLOT_HELPER = """\
static std::pair<std::vector<float>, int>& slot(std::pair<std::vector<float>, int>& p) {
    return p;
}
static std::vector<std::pair<float, float>>& rows(std::vector<std::pair<float, float>>& v) {
    return v;
}
"""


@pytest.mark.parametrize(
    ("body", "direct"),
    [
        (
            "    std::pair<std::vector<float>, int> p{std::vector<float>(n), 0};\n"
            "    auto& [values, status] = slot(p);\n"
            "    (void)status;\n"
            "    for (uint32_t i = 0; i < n; ++i) {\n"
            "        values[i] = af[i];\n"
            "    }\n"
            "    c = p.first;\n",
            ("slot(p)", "p"),
        ),
        (
            "    std::vector<std::pair<float, float>> pairs(n);\n"
            "    for (auto& [x, y] : rows(pairs)) {\n"
            "        x = af[0];\n"
            "        y = bf[0];\n"
            "    }\n"
            "    for (uint32_t i = 0; i < n; ++i) {\n"
            "        c[i] = pairs[i].first + pairs[i].second;\n"
            "    }\n",
            ("rows(pairs)", "pairs"),
        ),
    ],
    ids=["declaration", "range-for"],
)
def test_limit_a_write_through_a_reference_binding_of_a_call_is_missed(body: str, direct: tuple[str, str]) -> None:
    # A reference binding whose initializer or range has no base (a call) defines each name from the call's value
    # and links it with nothing, so a write through the name never reaches the object the call refers to; binding
    # the object itself reads True.
    assert_clear(read({"main.cpp": program(body, helpers=SLOT_HELPER)}))
    text = program(body.replace(*direct), helpers=SLOT_HELPER)
    assert_flow(read({"main.cpp": text}), text)


def test_limit_a_value_passed_through_an_exception_is_missed() -> None:
    # A catch parameter is declared with no value, so a thrown value never reaches it.
    body = "    try {\n        throw af[0];\n    } catch (float e) {\n        c[0] = e;\n    }\n"
    assert_clear(read({"main.cpp": program(READ_BACK + DEVICE_TO_C + body)}))


def test_limit_false_positive_one_object_holding_input_and_output() -> None:
    helpers = "struct Arrays {\n    std::vector<float> in;\n    std::vector<float> out;\n};\n"
    body = (
        "    Arrays s;\n"
        "    s.in.assign(af, af + n);\n"
        "    distributed::EnqueueWriteMeshBuffer(cq, in_buf, s.in, false);\n"
        + READ_BACK
        + "    s.out = res;\n"
    )
    tail = (
        "    uint64_t dims[1] = {n};\n"
        '    if (lassi_io_write(argv[3], "c", LASSI_IO_F32, 1u, dims, s.out.data()) != 0) {\n'
        "        return 1;\n"
        "    }\n"
    )
    text = program(body, helpers=helpers, tail=tail)
    assert_flow(read({"main.cpp": text}), text)
    # A reference structured binding links each name with the whole object, so golden values computed from the
    # inputs into one part reach the output read from the other; a value binding holds copies and reads False.
    run = "struct Run {\n    std::vector<float> out;\n    std::vector<float> expected;\n};\n"
    golden = (
        "    Run run;\n"
        "    auto& [out, expected] = run;\n"
        "    out.resize(n);\n"
        "    expected.resize(n);\n"
        "    distributed::EnqueueReadMeshBuffer(cq, out, c_buf, true);\n"
        "    for (uint32_t i = 0; i < n; ++i) {\n"
        "        expected[i] = af[i] + bf[i];\n"
        "    }\n"
        "    c = out;\n"
    )
    text = program(golden, helpers=run)
    assert_flow(read({"main.cpp": text}), text)
    assert_clear(read({"main.cpp": program(golden.replace("auto& [", "auto ["), helpers=run)}))


def test_limit_false_positive_a_pointer_re_pointed_from_input_to_output() -> None:
    body = (
        "    std::vector<float> staged(n);\n"
        "    float* p = staged.data();\n"
        "    for (uint32_t i = 0; i < n; ++i) {\n"
        "        p[i] = af[i];\n"
        "    }\n"
        "    distributed::EnqueueWriteMeshBuffer(cq, in_buf, staged, false);\n"
        + READ_BACK
        + "    p = c.data();\n"
        "    for (uint32_t i = 0; i < n; ++i) {\n"
        "        p[i] = res[i];\n"
        "    }\n"
    )
    text = program(body)
    assert_flow(read({"main.cpp": text}), text)


@pytest.mark.parametrize(
    "overwrite",
    [
        "    for (uint32_t i = 0; i < n; ++i) {\n        out[i] = res[i];\n    }\n",
        "    std::memcpy(out.data(), res.data(), n * sizeof(float));\n",
        "    std::copy(res.begin(), res.end(), out.begin());\n",
        "    out.assign(res.begin(), res.end());\n",
        "    out.swap(res);\n",
        "    auto& alias = out;\n    alias = res;\n",
    ],
    ids=["element-writes", "memcpy", "copy", "assign", "swap", "reference-assignment"],
)
def test_limit_false_positive_an_output_initialized_from_input_then_overwritten(overwrite: str) -> None:
    # Only a whole-variable assignment (`out = res;`, a strong update) or a device read into out or a name linked
    # with it clears the earlier value.
    start = "    std::vector<float> out(af, af + n);\n" + READ_BACK
    text = program(start + overwrite, tail=WRITE_OUT)
    assert_flow(read({"main.cpp": text}), text)
    assert_clear(read({"main.cpp": program(start + "    out = res;\n", tail=WRITE_OUT)}))


@pytest.mark.parametrize(
    "overwrite",
    [
        "    auto& [values, status] = p;\n    (void)status;\n    values = res;\n",
        "    auto put = [&held = p]() {\n        held = {res, 0};\n    };\n    put();\n",
    ],
    ids=["reference-binding-name", "reference-init-capture"],
)
def test_limit_false_positive_an_assignment_through_a_name_linked_with_the_object_is_a_weak_update(
    overwrite: str,
) -> None:
    # A reference structured-binding name and a reference init capture are linked with p, so a whole assignment
    # through either adds the device values to p's and keeps the staged input; assigning p itself replaces it.
    start = "    std::pair<std::vector<float>, int> p{std::vector<float>(af, af + n), 0};\n" + READ_BACK
    text = program(start + overwrite + "    c = p.first;\n")
    assert_flow(read({"main.cpp": text}), text)
    assert_clear(read({"main.cpp": program(start + "    p = {res, 0};\n    c = p.first;\n")}))


def test_limit_false_positive_a_refill_through_a_read_api_outside_the_set() -> None:
    body = (
        "    std::vector<float> staging(af, af + n);\n"
        "    EnqueueReadBuffer(cq, c_buf, staging, true);\n"
        "    c = staging;\n"
    )
    text = program(body)
    assert_flow(read({"main.cpp": text}), text)


def test_limit_false_positive_a_staging_write_after_the_read_in_a_loop_with_no_device_read_in_it() -> None:
    # The device read sits before the loop, so the loop's later staging write reaches the earlier read.
    body = (
        "    std::vector<float> buf(n);\n"
        "    distributed::EnqueueReadMeshBuffer(cq, buf, c_buf, true);\n"
        "    for (int pass = 0; pass < 1; ++pass) {\n"
        "        for (uint32_t i = 0; i < n; ++i) {\n"
        "            c[i] = buf[i];\n"
        "        }\n"
        "        for (uint32_t i = 0; i < n; ++i) {\n"
        "            buf[i] = af[i];\n"
        "        }\n"
        "        distributed::EnqueueWriteMeshBuffer(cq, in_buf, buf, false);\n"
        "    }\n"
    )
    text = program(body)
    assert_flow(read({"main.cpp": text}), text)


def test_limit_false_positive_every_conditional_branch_but_if_0_is_read() -> None:
    body = (
        "#ifndef LASSI_CPU_FALLBACK\n"
        + READ_INTO_C
        + "#else\n"
        "    for (uint32_t i = 0; i < n; ++i) {\n"
        "        c[i] = af[i] * 2.0f;\n"
        "    }\n"
        "#endif\n"
    )
    text = program(body)
    assert_flow(read({"main.cpp": text}), text)


def test_limit_a_device_read_in_a_member_function_clears_the_whole_object() -> None:
    # The object is one alias class (field-insensitive): reading device data into one member clears its input too.
    helpers = (
        "struct Prog {\n"
        "    lassi_io_array input = {};\n"
        "    std::vector<float> scratch;\n"
        "    std::vector<float> out;\n"
        "    void load(const char* path) {\n"
        "        lassi_io_read(path, &input);\n"
        "    }\n"
        "    void fetch(distributed::MeshCommandQueue& queue, std::shared_ptr<distributed::MeshBuffer>& buffer) {\n"
        "        distributed::EnqueueReadMeshBuffer(queue, scratch, buffer, true);\n"
        "    }\n"
        "    void host() {\n"
        "        const float* x = static_cast<const float*>(input.data);\n"
        "        out.resize(input.count);\n"
        "        for (uint64_t i = 0; i < input.count; ++i) {\n"
        "            out[i] = x[i] + 1.0f;\n"
        "        }\n"
        "    }\n"
        "};\n"
    )
    body = "    Prog p;\n    p.load(argv[1]);\n    p.fetch(cq, c_buf);\n    p.host();\n"
    tail = '    uint64_t dims[1] = {n};\n    lassi_io_write(argv[3], "c", LASSI_IO_F32, 1u, dims, p.out.data());\n'
    assert_clear(read({"main.cpp": program(body, helpers=helpers, tail=tail)}))


ARRAYS = "struct Arrays {\n    std::vector<float> in;\n    std::vector<float> out;\n};\n"


@pytest.mark.parametrize(
    ("body", "destination"),
    [
        (
            "    Arrays s;\n"
            "    s.in.assign(af, af + n);\n"
            "    s.out.resize(n);\n"
            "    distributed::EnqueueReadMeshBuffer(cq, s.out, c_buf, true);\n"
            "    c = s.in;\n",
            "s.out",
        ),
        (
            "    Arrays s;\n"
            "    s.in.assign(af, af + n);\n"
            "    auto& o = s.out;\n"
            "    distributed::EnqueueReadMeshBuffer(cq, o, c_buf, true);\n"
            "    c = s.in;\n",
            "o",
        ),
        (
            "    std::pair<std::vector<float>, std::vector<float>> io{std::vector<float>(af, af + n), "
            "std::vector<float>(n)};\n"
            "    auto& [in_vals, out_vals] = io;\n"
            "    distributed::EnqueueReadMeshBuffer(cq, out_vals, c_buf, true);\n"
            "    c = in_vals;\n",
            "out_vals",
        ),
        (
            "    Arrays s;\n"
            "    s.in.assign(af, af + n);\n"
            "    [&, &o = s.out]() {\n"
            "        distributed::EnqueueReadMeshBuffer(cq, o, c_buf, true);\n"
            "    }();\n"
            "    c = s.in;\n",
            "o",
        ),
        (
            "    Arrays s;\n"
            "    s.in.assign(af, af + n);\n"
            "    auto fetch_out = [&](std::vector<float>& o) {\n"
            "        distributed::EnqueueReadMeshBuffer(cq, o, c_buf, true);\n"
            "    };\n"
            "    fetch_out(s.out);\n"
            "    c = s.in;\n",
            "o",
        ),
    ],
    ids=[
        "member",
        "reference-to-a-member",
        "reference-binding-name",
        "reference-init-capture",
        "named-lambda-linked-parameter",
    ],
)
def test_limit_a_device_read_into_one_part_of_an_object_clears_the_whole_object(body: str, destination: str) -> None:
    # The device read kills the alias class of its operand's base, the whole object, so the input part copied to the
    # output afterwards reads False; the same copy after a device read into res reads True.
    assert_clear(read({"main.cpp": program(body, helpers=ARRAYS)}))
    text = program(body.replace(f"(cq, {destination},", "(cq, res,"), helpers=ARRAYS)
    assert_flow(read({"main.cpp": text}), text)


NO_LINK_HELPERS = (
    FIRST_OF
    + SLOT_HELPER
    + "static std::vector<std::vector<float>>& all_of(std::vector<std::vector<float>>& v) {\n"
    "    return v;\n"
    "}\n"
    "static std::vector<std::pair<std::vector<float>, int>>& all_slots(\n"
    "    std::vector<std::pair<std::vector<float>, int>>& v) {\n"
    "    return v;\n"
    "}\n"
)


@pytest.mark.parametrize(
    ("body", "named"),
    [
        (
            "    std::vector<float> staged(af, af + n);\n"
            "    auto&& r = std::move(staged);\n"
            "    distributed::EnqueueReadMeshBuffer(cq, r, c_buf, true);\n"
            "    c = staged;\n",
            ("(cq, r,", "(cq, staged,"),
        ),
        (
            "    std::vector<std::vector<float>> outs;\n"
            "    outs.emplace_back(af, af + n);\n"
            "    std::ranges::for_each(outs, [&](std::vector<float>& v) {\n"
            "        distributed::EnqueueReadMeshBuffer(cq, v, c_buf, true);\n"
            "    });\n"
            "    c = outs[0];\n",
            ("std::ranges::for_each(outs,", "std::for_each(outs.begin(), outs.end(),"),
        ),
        (
            "    std::vector<float> staged(af, af + n);\n"
            "    std::invoke([&](std::vector<float>& v) {\n"
            "        distributed::EnqueueReadMeshBuffer(cq, v, c_buf, true);\n"
            "    }, staged);\n"
            "    c = staged;\n",
            ("(cq, v,", "(cq, staged,"),
        ),
        (
            "    std::vector<float> staged(af, af + n);\n"
            "    float* p = first_of(staged);\n"
            "    ReadFromBuffer(*c_buf, reinterpret_cast<uint8_t*>(p));\n"
            "    c = staged;\n",
            ("first_of(staged)", "staged.data()"),
        ),
        (
            "    std::vector<std::vector<float>> outs;\n"
            "    outs.emplace_back(af, af + n);\n"
            "    for (auto& v : all_of(outs)) {\n"
            "        distributed::EnqueueReadMeshBuffer(cq, v, c_buf, true);\n"
            "    }\n"
            "    c = outs[0];\n",
            ("all_of(outs)", "outs"),
        ),
        (
            "    std::vector<float> staged(af, af + n);\n"
            "    std::vector<float> spare(n);\n"
            "    auto& dst = n > 0 ? staged : spare;\n"
            "    distributed::EnqueueReadMeshBuffer(cq, dst, c_buf, true);\n"
            "    c = staged;\n",
            ("n > 0 ? staged : spare", "staged"),
        ),
        (
            "    std::vector<std::vector<float>> outs;\n"
            "    outs.emplace_back(af, af + n);\n"
            "    auto get = [&](std::vector<float>& v) {\n"
            "        distributed::EnqueueReadMeshBuffer(cq, v, c_buf, true);\n"
            "    };\n"
            "    std::for_each(outs.begin(), outs.end(), get);\n"
            "    c = outs[0];\n",
            ("std::for_each(outs.begin(), outs.end(), get);", "get(outs[0]);"),
        ),
        (
            "    std::pair<std::vector<float>, int> p{std::vector<float>(af, af + n), 0};\n"
            "    auto& [values, status] = slot(p);\n"
            "    (void)status;\n"
            "    distributed::EnqueueReadMeshBuffer(cq, values, c_buf, true);\n"
            "    c = p.first;\n",
            ("slot(p)", "p"),
        ),
        (
            "    std::vector<std::pair<std::vector<float>, int>> ps;\n"
            "    ps.emplace_back(std::vector<float>(af, af + n), 0);\n"
            "    for (auto& [values, status] : all_slots(ps)) {\n"
            "        (void)status;\n"
            "        distributed::EnqueueReadMeshBuffer(cq, values, c_buf, true);\n"
            "    }\n"
            "    c = ps[0].first;\n",
            ("all_slots(ps)", "ps"),
        ),
        (
            "    std::pair<std::vector<float>, int> p{std::vector<float>(af, af + n), 0};\n"
            "    std::pair<std::vector<float>, int> q{std::vector<float>(n), 0};\n"
            "    auto& [values, status] = n > 0 ? p : q;\n"
            "    (void)status;\n"
            "    distributed::EnqueueReadMeshBuffer(cq, values, c_buf, true);\n"
            "    c = p.first;\n",
            ("n > 0 ? p : q", "p"),
        ),
    ],
    ids=[
        "reference-bound-to-a-call",
        "ranges-for-each-parameter",
        "invoked-lambda-parameter",
        "pointer-bound-to-a-call",
        "range-for-element-over-a-call",
        "reference-bound-to-a-conditional",
        "lambda-variable-given-to-for-each",
        "reference-binding-of-a-call",
        "range-for-reference-binding-over-a-call",
        "reference-binding-of-a-conditional",
    ],
)
def test_limit_false_positive_a_device_read_through_a_name_with_no_alias_link_kills_only_that_name(
    body: str, named: tuple[str, str]
) -> None:
    # The name has no link to the object it refers to, so the device read leaves the staged input in the object;
    # naming the object (or a std::for_each parameter, which is linked) reads False. The named lambda whose calls
    # pass different objects is pinned by
    # test_limit_a_named_lambda_parameter_whose_calls_pass_different_objects_is_not_linked, and the one called inside
    # another named lambda by
    # test_limit_false_positive_a_device_read_through_a_named_lambda_called_inside_another_kills_only_that_name.
    text = program(body, helpers=NO_LINK_HELPERS)
    assert_flow(read({"main.cpp": text}), text)
    assert_clear(read({"main.cpp": program(body.replace(*named), helpers=NO_LINK_HELPERS)}))


def test_limit_false_positive_same_named_methods_are_united() -> None:
    helpers = (
        "struct Mirror {\n"
        "    void put(float* y, const float* x, uint32_t count) {\n"
        "        for (uint32_t i = 0; i < count; ++i) {\n"
        "            y[i] = x[i];\n"
        "        }\n"
        "    }\n"
        "};\n"
        "struct Zero {\n"
        "    void put(float* y, const float* x, uint32_t count) {\n"
        "        (void)x;\n"
        "        for (uint32_t i = 0; i < count; ++i) {\n"
        "            y[i] = 0.0f;\n"
        "        }\n"
        "    }\n"
        "};\n"
    )
    text = program("    Zero z;\n    z.put(c.data(), af, n);\n", helpers=helpers)
    assert_flow(read({"main.cpp": text}), text)
    # Free-function overloads of one arity are united the same way: C++ calls the float one, which writes zeros.
    overloads = (
        "static void put(float* y, const float* x, uint32_t count) {\n"
        "    (void)x;\n"
        "    for (uint32_t i = 0; i < count; ++i) {\n"
        "        y[i] = 0.0f;\n"
        "    }\n"
        "}\n"
        "static void put(double* y, const double* x, uint32_t count) {\n"
        "    for (uint32_t i = 0; i < count; ++i) {\n"
        "        y[i] = x[i];\n"
        "    }\n"
        "}\n"
    )
    text = program("    put(c.data(), af, n);\n", helpers=overloads)
    assert_flow(read({"main.cpp": text}), text)


def test_limit_false_positive_every_constructor_argument_flows_into_the_object() -> None:
    # Member initializers are not read, so an input-derived count given to a constructor makes the object carry
    # input, and a member written later is a weak update of that one object.
    helpers = "struct Box {\n    std::vector<float> v;\n    explicit Box(uint32_t count) : v(count) {}\n};\n"
    body = READ_BACK + "    Box box(static_cast<uint32_t>(af[0]));\n    box.v = res;\n    c = box.v;\n"
    text = program(body, helpers=helpers)
    assert_flow(read({"main.cpp": text}), text)


def test_limit_false_positive_an_undeclared_name_in_a_class_with_an_unknown_base_is_a_member() -> None:
    # FLT_MAX comes from a system header the guard never reads; with a base outside the host text it reads as THIS.
    open_class = "class Runner : public RunnerBase {"
    helpers = RUNNER_WITH_EXPR.replace("EXPR", "FLT_MAX").replace("class Runner {", open_class)
    body = "    Runner r;\n    r.load(af, n);\n" + READ_BACK + "    r.finish(res, c);\n"
    text = program(body, helpers=helpers)
    assert_flow(read({"main.cpp": text}), text)


def test_limit_false_positive_a_lambda_not_called_by_name_is_read_where_it_is_written() -> None:
    body = (
        "    std::vector<float> host(n);\n"
        "    for (uint32_t i = 0; i < n; ++i) {\n"
        "        host[i] = af[i];\n"
        "    }\n"
        "    distributed::EnqueueWriteMeshBuffer(cq, in_buf, host, false);\n"
        "    auto unpack = [&]() {\n"
        "        for (uint32_t i = 0; i < n; ++i) {\n"
        "            c[i] = host[i];\n"
        "        }\n"
        "    };\n"
        "    distributed::EnqueueReadMeshBuffer(cq, host, c_buf, true);\n"
        "    std::invoke(unpack);\n"
    )
    text = program(body)
    assert_flow(read({"main.cpp": text}), text)
    # A device read in such a lambda written before the staging is read before it, so the staged input stays in the
    # buffer; written after the staging, it clears it.
    invoke = "    std::invoke(fetch);\n"
    text = program(BUF_ONLY + FETCH_BUF + STAGE_BUF_LOOP + invoke + BUF_TO_C)
    assert_flow(read({"main.cpp": text}), text)
    assert_clear(read({"main.cpp": program(BUF_ONLY + STAGE_BUF_LOOP + FETCH_BUF + invoke + BUF_TO_C)}))


# A buffer staged from the input by a loop, a lambda that reads the device result back into it, and its copy into c.
BUF_ONLY = "    std::vector<float> buf(n);\n"
FETCH_BUF = "    auto fetch = [&]() {\n        distributed::EnqueueReadMeshBuffer(cq, buf, c_buf, true);\n    };\n"
STAGE_BUF_LOOP = (
    "    for (uint32_t i = 0; i < n; ++i) {\n"
    "        buf[i] = af[i];\n"
    "    }\n"
    "    distributed::EnqueueWriteMeshBuffer(cq, in_buf, buf, false);\n"
)
BUF_TO_C = "    for (uint32_t i = 0; i < n; ++i) {\n        c[i] = buf[i];\n    }\n"


def test_limit_false_positive_a_named_lambda_called_inside_another_named_lambda_is_read_where_it_is_written() -> None:
    # copy_out is called inside finish's body, so it stays where it is written, before the device read, and its [&]
    # capture of buf reads the staged input there; called after finish() instead, it runs at its call (False).
    copy_out = (
        "    auto copy_out = [&]() {\n"
        "        for (uint32_t i = 0; i < n; ++i) {\n"
        "            c[i] = buf[i];\n"
        "        }\n"
        "    };\n"
    )
    text = program(STAGE_BUF + copy_out + finish_with("copy_out();"))
    assert_flow(read({"main.cpp": text}), text)
    after = finish_with("(void)0;") + "    copy_out();\n"
    assert_clear(read({"main.cpp": program(STAGE_BUF + copy_out + after)}))
    # The same holds only between the staging and the read-back: convert, called inside run after run's read-back,
    # reads the staged input when it is written after the staging (True), but written before the staging it is read
    # there, where buf holds no input yet (False).
    run = run_with(READ_BACK_BUF + "    convert();\n")
    text = program(BUF_ONLY + STAGE_BUF_LOOP + CONVERT_BUF + run)
    assert_flow(read({"main.cpp": text}), text)
    assert_clear(read({"main.cpp": program(BUF_ONLY + CONVERT_BUF + STAGE_BUF_LOOP + run)}))
    # fetch, written before the staging and called only inside run, does its device read where it is written, before
    # the staging, so the staged input stays in buf; called directly after the staging, it runs there (False).
    launch = "    distributed::EnqueueMeshWorkload(cq, workload, false);\n"
    text = program(BUF_ONLY + FETCH_BUF + STAGE_BUF_LOOP + run_with(launch + "    fetch();\n") + BUF_TO_C)
    assert_flow(read({"main.cpp": text}), text)
    assert_clear(read({"main.cpp": program(BUF_ONLY + FETCH_BUF + STAGE_BUF_LOOP + "    fetch();\n" + BUF_TO_C)}))


RUN_CALL = "    run();\n"
CONVERT_BUF = (
    "    auto convert = [&]() {\n"
    "        for (uint32_t i = 0; i < n; ++i) {\n"
    "            c[i] = buf[i];\n"
    "        }\n"
    "    };\n"
)
READ_BACK_BUF = "    distributed::EnqueueReadMeshBuffer(cq, buf, c_buf, true);\n"


def run_defined(calls: str) -> str:
    """Return run_with(calls) without run's call: only the definition of the named lambda run."""
    return run_with(calls).removesuffix(RUN_CALL)


@pytest.mark.parametrize(
    ("helper", "calls", "order"),
    [
        (FETCH_BUF, "    fetch();\n", "before-staging"),
        (CONVERT_BUF, "    convert();\n", "before-read-back"),
        (FETCH_BUF, "    std::invoke(fetch);\n", "before-staging"),
        (CONVERT_BUF, "    std::invoke(convert);\n", "before-read-back"),
    ],
    ids=["device-read-lambda", "conversion-lambda", "device-read-lambda-not-called", "conversion-lambda-not-called"],
)
def test_limit_false_positive_a_lambda_written_inside_a_named_lambda_is_read_with_its_body_at_its_calls(
    helper: str, calls: str, order: str
) -> None:
    # run is written before the staging (or the read-back) and called after it. The helper written just before run,
    # outside any named lambda, is read there, before the staging or the read-back: the false positive reads True.
    # Written inside run's body, it is read with that body at run's call, after the staging or the read-back, and
    # reads False, whether a call names it or std::invoke runs it.
    def text(definitions: str) -> str:
        if order == "before-staging":
            return program(BUF_ONLY + definitions + STAGE_BUF_LOOP + RUN_CALL + BUF_TO_C)
        return program(BUF_ONLY + STAGE_BUF_LOOP + definitions + READ_BACK_BUF + RUN_CALL)

    outside = text(helper + run_defined(calls))
    assert_flow(read({"main.cpp": outside}), outside)
    assert_clear(read({"main.cpp": text(run_defined(helper + calls))}))


FETCH_BUF_THROUGH_PARAMETER = (
    "    auto fetch = [&](std::vector<float>& v) {\n"
    "        distributed::EnqueueReadMeshBuffer(cq, v, c_buf, true);\n"
    "    };\n"
    "    fetch(buf);\n"
)
FETCH_BUF_THROUGH_INIT_CAPTURE = (
    "    auto fetch = [&, &v = buf]() {\n"
    "        distributed::EnqueueReadMeshBuffer(cq, v, c_buf, true);\n"
    "    };\n"
    "    fetch();\n"
)


@pytest.mark.parametrize(
    ("calls", "tail"),
    [
        (CONVERT_BUF + READ_BACK_BUF + "    convert();\n", ""),
        (FETCH_BUF_THROUGH_PARAMETER, BUF_TO_C),
        (FETCH_BUF_THROUGH_INIT_CAPTURE, BUF_TO_C),
    ],
    ids=["conversion-lambda", "reference-parameter", "reference-init-capture"],
)
@pytest.mark.parametrize(
    "runner",
    ["    std::invoke(run);\n", "    std::vector<std::function<void()>> jobs{run};\n    jobs[0]();\n"],
    ids=["std-invoke", "std-function-vector"],
)
def test_limit_false_positive_a_named_lambda_encloses_one_called_inside_it_only_when_a_call_names_it(
    calls: str, tail: str, runner: str
) -> None:
    # run, written after the staging, holds a named lambda and that lambda's call. Called by name, run encloses it,
    # so the inner lambda is read with run's body at run's call, where it is written in that body: the conversion
    # lambda before run's read-back, and the device read through the parameter or the reference init capture
    # refills nothing, so the staged input in buf reaches c and the false positive reads True. When no call names
    # run (std::invoke runs it, or a std::function in a vector), run encloses nothing: the inner lambda runs at its
    # call, after the read-back or with its parameter or capture linked with buf, and reads False.
    by_name = program(BUF_ONLY + STAGE_BUF_LOOP + run_with(calls) + tail)
    assert_flow(read({"main.cpp": by_name}), by_name)
    assert_clear(read({"main.cpp": program(BUF_ONLY + STAGE_BUF_LOOP + run_defined(calls) + runner + tail)}))


def test_limit_false_positive_kernels_created_outside_the_creator_set() -> None:
    kernels = "    LightMetalReplay replay{};\n    (void)replay;\n"
    text = program(CLEAN_BODY, kernels=kernels)
    assert_no_kernel(read({"main.cpp": text}), text)


@pytest.mark.parametrize(
    ("echo", "sink"),
    [
        ("    std::fwrite(af, sizeof(float), n, stdout);\n", "std::fwrite(af"),
        ("    ::write(2, af, 4 * n);\n", "::write(2"),
        (
            '    std::ofstream log("input.log", std::ios::binary);\n'
            "    log.write(reinterpret_cast<const char*>(af), static_cast<std::streamsize>(4 * n));\n",
            "log.write(",
        ),
    ],
    ids=["fwrite-stdout", "posix-write-stderr", "stream-log-file"],
)
def test_limit_false_positive_input_echoed_through_a_listed_write(echo: str, sink: str) -> None:
    # Every listed write is an output write wherever it writes, so an echo of an input reads as host compute.
    text = program(CLEAN_BODY + echo)
    assert_flow(read({"main.cpp": text}), text, sink=sink)


CPU_REFERENCE_MAIN = (
    SYNTHETIC_LINE
    + "\n"
    + """\
#include "lassi_io.h"
int main(int argc, char** argv) {
    lassi_io_array a = {0};
    lassi_io_array b = {0};
    lassi_io_read(argv[1], &a);
    lassi_io_read(argv[2], &b);
    const float* x = (const float*)a.data;
    const float* y = (const float*)b.data;
    float z[4];
    for (int i = 0; i < 4; ++i) {
        z[i] = x[i] + y[i];
    }
    uint64_t dims[1] = {4};
    lassi_io_write(argv[3], "c", LASSI_IO_F32, 1u, dims, z);
    return 0;
}
"""
)


@pytest.mark.parametrize("path", ["ref.c", "cpu_reference.h"])
def test_limit_false_positive_a_main_in_a_file_the_build_never_compiles(path: str) -> None:
    # ttmetal-host compiles only .cpp, .cc, and .cxx files outside kernels/, but every model file with a host suffix
    # is a root, and every free main outside a namespace is checked beside the program's own.
    assert_clear(read({"main.cpp": program(CLEAN_BODY)}))
    result = read({"main.cpp": program(CLEAN_BODY), path: CPU_REFERENCE_MAIN})
    assert_flow(result, CPU_REFERENCE_MAIN, source="lassi_io_read(argv[1]", sink="lassi_io_write(", file=path)


STAGE_FILE = (
    SYNTHETIC_LINE
    + "\n"
    + """\
#include <cstdint>
#include <vector>
static std::vector<float> buf;
void stage(const float* x, uint32_t count) {
    buf.assign(x, x + count);
}
"""
)


def test_limit_false_positive_same_named_file_scope_variables_are_one_symbol() -> None:
    # One symbol per file-scope name: namespaced variables, and file-static ones in two files, are united.
    namespaces = "namespace staging {\nstd::vector<float> data;\n}\nnamespace result {\nstd::vector<float> data;\n}\n"
    body = (
        "    staging::data.assign(af, af + n);\n"
        "    distributed::EnqueueWriteMeshBuffer(cq, in_buf, staging::data, false);\n"
        + READ_BACK
        + "    result::data = res;\n"
        "    c = result::data;\n"
    )
    text = program(body, helpers=namespaces)
    assert_flow(read({"main.cpp": text}), text)
    helpers = "void stage(const float* x, uint32_t count);\nstatic std::vector<float> buf;\n"
    body = "    stage(af, n);\n" + READ_BACK + "    buf.assign(res.begin(), res.end());\n    c = buf;\n"
    text = program(body, helpers=helpers)
    assert_flow(read({"main.cpp": text, "stage.cpp": STAGE_FILE}), text)
    assert_clear(read({"main.cpp": text, "stage.cpp": STAGE_FILE.replace("buf", "staged")}))


def test_limit_false_positive_a_device_read_into_a_buffer_named_through_a_macro_kills_nothing() -> None:
    body = (
        "    for (uint32_t i = 0; i < n; ++i) {\n"
        "        res[i] = af[i];\n"
        "    }\n"
        "    distributed::EnqueueWriteMeshBuffer(cq, in_buf, res, false);\n"
        "    distributed::EnqueueReadMeshBuffer(cq, RESULT, c_buf, true);\n"
        + DEVICE_TO_C
    )
    text = program(body, helpers="#define RESULT res\n")
    assert_flow(read({"main.cpp": text}), text)
    assert_clear(read({"main.cpp": program(body.replace("RESULT", "res"))}))


MATRIX = """\
struct Matrix {
    uint32_t rows = 0;
    std::vector<float> values;
};
static Matrix load(const float* p, uint32_t count) {
    Matrix m;
    m.rows = count;
    m.values.assign(p, p + count);
    return m;
}
"""
LOAD_A = "    Matrix A = load(af, n);\n    distributed::EnqueueWriteMeshBuffer(cq, in_buf, A.values, false);\n"
STAGE_WITH_SCALE = """\
struct Stage {
    std::vector<float> staged;
    float scale = 1.0f;
    void load(const float* p, uint32_t count) {
        staged.assign(p, p + count);
    }
    float convert(float v) const {
        return v * scale;
    }
};
"""


@pytest.mark.parametrize(
    ("helpers", "body", "tail"),
    [
        (
            MATRIX,
            LOAD_A + READ_BACK + "    Matrix C;\n    C.rows = A.rows;\n    C.values = res;\n",
            '    uint64_t dims[1] = {n};\n    lassi_io_write(argv[3], "c", LASSI_IO_F32, 1u, dims, C.values.data());\n',
        ),
        (MATRIX, LOAD_A + READ_BACK + "    res = untilize_nfaces(res, A.rows, 1u);\n" + DEVICE_TO_C, WRITE_C),
        (
            MATRIX,
            LOAD_A + READ_BACK + "    res = untilize_nfaces(res, A.values.size(), 1u);\n" + DEVICE_TO_C,
            WRITE_C,
        ),
        (
            MATRIX,
            "    auto [rows, values] = load(af, n);\n"
            "    distributed::EnqueueWriteMeshBuffer(cq, in_buf, values, false);\n"
            + READ_BACK
            + "    res = untilize_nfaces(res, rows, 1u);\n"
            + DEVICE_TO_C,
            WRITE_C,
        ),
        (
            MATRIX,
            "    std::vector<Matrix> parts{load(af, n)};\n"
            "    for (const auto& [rows, values] : parts) {\n"
            "        distributed::EnqueueWriteMeshBuffer(cq, in_buf, values, false);\n"
            "        distributed::EnqueueReadMeshBuffer(cq, res, c_buf, true);\n"
            "        res = untilize_nfaces(res, rows, 1u);\n"
            "    }\n" + DEVICE_TO_C,
            WRITE_C,
        ),
        (
            STAGE_WITH_SCALE,
            "    Stage s;\n"
            "    s.load(af, n);\n"
            "    distributed::EnqueueWriteMeshBuffer(cq, in_buf, s.staged, false);\n"
            + READ_BACK
            + "    for (uint32_t i = 0; i < n; ++i) {\n        c[i] = s.convert(res[i]);\n    }\n",
            WRITE_C,
        ),
        (
            "",
            "    std::pair<uint32_t, std::vector<float>> A{n, std::vector<float>(af, af + n)};\n"
            "    distributed::EnqueueWriteMeshBuffer(cq, in_buf, A.second, false);\n"
            + READ_BACK
            + "    res = untilize_nfaces(res, A.first, 1u);\n"
            + DEVICE_TO_C,
            WRITE_C,
        ),
    ],
    ids=[
        "shape-copied-to-the-output-object",
        "shape-passed-to-a-library-call",
        "member-container-size",
        "structured-binding",
        "range-for-structured-binding",
        "scalar-member",
        "library-pair-member",
    ],
)
def test_limit_false_positive_a_member_or_binding_of_an_input_object_carries_all_its_input(
    helpers: str, body: str, tail: str
) -> None:
    # Only a METADATA member (a.dims, a.count) of an object is no mention of it; any other member (A.values in
    # A.values.size() included, and the first of a std::pair), and a structured binding's name, read as the whole
    # object, initializer, or range, input values included.
    text = program(body, helpers=helpers, tail=tail)
    assert_flow(read({"main.cpp": text}), text)
    shape_free = body.replace("A.rows", "n").replace("A.values.size()", "n").replace("A.first", "n")
    shape_free = shape_free.replace("untilize_nfaces(res, rows,", "untilize_nfaces(res, n,")
    assert_clear(read({"main.cpp": program(shape_free, helpers=helpers.replace("v * scale", "v"), tail=tail)}))
