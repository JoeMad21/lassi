"""The CPU -> TT host-compute guard of the Harness Contract (task P4.12): read_host_compute.

The guard reads only the host text of a built TT host program: the model's
host files and every file they include. It never reads the run, its
streams, the kernel JIT cache, the simulator's files, or any output file,
so nothing a program prints or writes can change its reading. It starts no
process and opens no file: it reads only its two arguments. ttmetal-host
declares the Toolchain capability host_compute_guard
(lassi.core.capabilities) and answers with read_host_compute;
lassi.core.stages asks for the reading after every build that gave a
program, before any run.

Host text. The roots are the model's files whose suffix is one of
HOST_SUFFIXES and that are not kernel sources
(lassi.toolchains.ttmetal_build.is_kernel_source: a directory named
kernels in the path). Every `#include "N"` or `#include <N>` outside an
`#if 0` or `#if false` region in a file read is followed
(lassi.toolchains._cxx_scan resolve_include): a quoted name beside the
including file first, then at the build-directory root; an angle name at
the root only; the model's files before the harness files at each
candidate. A hit is read whatever its suffix or directory, kernels/
included, since `-idirafter .` lets host code include any file of the
build directory. Harness files and kernel sources that nothing includes
are never read, and a name that leaves the build directory or names a
pinned or system header is not read. Of the roots, ttmetal-host compiles
only the .cpp, .cc, and .cxx files; the others (a .c file, a header) are
read whether or not anything includes them. The roots are read in sorted
path order and the files they include after them, breadth-first in include
order; the host text is returned in sorted path order, so the reading
does not depend on the order of either mapping. The text is unreadable
when, in that read order, a file brings the UTF-8 bytes read past
SIZE_LIMIT (4194304) or holds an unterminated block comment, an
unterminated string, char, or raw-string literal in code (in a directive
line or an `#if 0` region such a quote only ends the line), or more than
256 brackets open at once in its code; the first such file gives the
reason. It is also unreadable when the reading needs more than STEP_LIMIT
(5000000) reader steps (lassi.toolchains._cxx_scan Budget). The reasons
read "more than 4194304 bytes of host text", "an unterminated comment in
<path>", "an unterminated literal in <path>", "brackets nested deeper than
256 in <path>", and "the reader exceeded its budget of 5000000 steps".

The identifier sets are facts of tt-metal 5280a9cf, read from the pinned
API headers under tt_metal/api/tt-metalium
(plans/spikes/p4-guard-identifiers.md: rx 20261004-162204-exec-29e2, rx
20261004-162216-exec-362c, and rx 20261004-162228-exec-16ba); a pin change
re-checks them (Agent Rule 10):

- KERNEL_CREATORS: CreateKernel (host_api.hpp:164, and a Quasar overload at
  experimental/host_api.hpp:59), CreateKernelFromString (host_api.hpp:184),
  KernelDescriptor (program_descriptors.hpp:101), and ProgramDescriptor
  (program_descriptors.hpp:136; Program(const ProgramDescriptor&) at
  program.hpp:29).
- COMPUTE_CONFIGS: ComputeConfig (kernel_types.hpp:76) and
  ComputeConfigDescriptor (program_descriptors.hpp:85).
- DEVICE_READS: EnqueueReadMeshBuffer (distributed.hpp:80), ReadShard
  (distributed.hpp:51, and tt_metal.hpp:123 and :136), ReadFromBuffer
  (tt_metal.hpp:100, :112, and :119), ReadFromDeviceL1 (tt_metal.hpp:351
  and :374), ReadFromDeviceDRAMChannel (tt_metal.hpp:275 and :292), and
  ReadRegFromDevice (tt_metal.hpp:382).
  No argument position is assumed: each argument that is a buffer operand
  (`res`, `c.data()`, `&c[0]`, `*p`) has its base's alias class killed,
  and any other argument (`a != nullptr`, `true`, `n * 4`) kills nothing
  (lassi.toolchains._cxx_flow, Device reads).

The rules. A name counts when it is an identifier token in code or in a
directive line other than an `#include` line (a header name is no
identifier), never in a comment, a literal, or an `#if 0` or `#if false`
region.

- creates_kernel: a KERNEL_CREATORS name appears in the host text.
  creates_compute: a COMPUTE_CONFIGS name appears. writes_output: a
  function body in the host text holds an output write (lassi_io_write
  with 6 arguments, fwrite with 4, `X.write(p, n)` where base(X) is a
  variable or a parameter whose declared type names ofstream, fstream, or
  ostream, or POSIX write with 3; lassi.toolchains._cxx_flow, Events),
  outside the `lassi_io_*` definitions, which are never read.
- Rule K: writes_output and not creates_kernel. No device kernel exists to
  compute any output value, so the program computes on the host.
- Rule T, the data-movement-only tag: creates_kernel and not
  creates_compute. A kernel's kind does not show where values are computed,
  so such a program is tagged, never failed.
- Rule H: values read from the inputs reach an output write through host
  code without passing through a device read (lassi.toolchains._cxx_flow,
  with DEVICE_READS as the device reads).

The reading, always a HostComputeReading:

1. Unreadable host text: None and one guard-not-checked note naming why.
2. No output write: None, the tag when rule T fires, and a
   guard-not-checked note.
3. Rule K: True and one guard-host-compute note at the first output write
   (by sorted path, then position), with no tag.
4. Rule H: True, the tag when rule T fires, and one guard-host-compute note
   at the witness's output write naming the input read and the output
   write as <file>:<line>.
5. Otherwise False and the tag when rule T fires.

Any other Exception in these steps gives None and a guard-not-checked note
naming the exception's class; the run goes on and the defect stays visible
in the record. An interrupt or a system exit is never caught.

Every Diagnostic is parse-stage (lassi.core.capabilities GUARD_STAGE): the
tag is a warning, the rest are notes, and every code starts with "guard-".
The tag sits at the first kernel-creator token, by position, of the first
model file in sorted path order that names one (of the first such harness
file when no model file does). A diagnostic's column is not set. Messages
are plain ASCII: a character of a message outside printable ASCII, such as
one in a file name, is written as a backslash escape, while the
Diagnostic's `file` keeps the path as given.

Limits. These are the known cases, found by review, not a complete
account: the reader approximates C++ statically, so other constructs can
read either way. The test_limit_* tests of
tests/toolchains/test_ttmetal_guard.py pin each reading below except the
misparses.

Misses, which read False (or not checked):
- host-computed values laundered through a device buffer;
- dead code: kills follow text order, so a device read in dead code
  between a host write and the output write clears it, and a kernel
  creator or a compute config in dead code counts;
- loops made with goto: only for, while, and do statements have back
  edges, so a value that reaches the output only around a goto's back
  edge is missed;
- kernels created but never launched or doing no work;
- values that reach an output only through subscripts, branch conditions,
  loop bounds, counts (a container's size or count argument, resize), or
  input metadata (a member in METADATA, such as a.count or a.dims[0]);
- macros: function-like macros, token pasting, and object-like macros, of
  which only a one-identifier alias in an unqualified free callee's name
  is expanded (one level), so a variable, a sink, or a member callee named
  through a macro is missed;
- calls through function pointers, and calls of an object's operator()
  (a functor called by its name, as a temporary, or as
  `x.operator()(...)`), which match no own definition;
- member functions of a class defined inside a function body, which are
  never read;
- library calls outside the standard-call table of
  lassi.toolchains._cxx_flow that write through a pointer or iterator
  argument (std::merge, std::rotate_copy, std::uninitialized_copy, an
  unknown call; std::back_inserter as a copy target included);
  std::ranges::for_each and std::invoke, whose lambda's parameter is never
  bound; and std::for_each or std::for_each_n given a lambda variable
  rather than a lambda expression, which binds nothing;
- a pointer or reference bound to a call, such as std::move(x) or a helper
  that returns a reference or v.data(), and a range-for reference element
  over a call: a call has no base, so the name is linked with nothing and
  holds no value, and a write or a read through it is missed; and a
  pointer, a reference, or a reference structured-binding name bound to a
  conditional, whose base, if any, is its first mention, a name in the
  condition when the condition names one: bound to `n > 0 ? p : q`, it is
  linked with n, not with p or q, so a write or a read through it misses
  them;
- a write through an own function's pointer, reference, or view parameter
  whose argument has no base, such as a call (`pick(c)`) or a view
  temporary (`std::span<float>(c)`, `std::span<float>{c}`): the parameter
  binds to no object of the caller, so the write reaches none;
- a value a lambda returns from its captures or holds in a value init
  capture: a lambda's return defines no RET, a call through a lambda
  variable is no own call and contributes only its arguments, and a value
  init capture `[x = v]` declares x in the lambda with no value (a
  reference init capture `[&x = v]` is linked with base(v), as a
  reference declaration is, except in a named lambda called inside
  another, below);
- a write through a named lambda's pointer or reference parameter (a
  std::function variable initialized with a lambda included) whose calls
  do not all pass it arguments with one same base (two calls with
  different objects, or an argument with no base): such a parameter is
  defined from each call's argument, never linked to it;
- a value passed in or out through a parameter or a reference init
  capture of a named lambda called inside another: one that a call names
  and that is written inside the body of another named lambda that a call
  names, or that has a call inside the body of a named lambda that a call
  names, its own included (lassi.toolchains._cxx_flow, Positions). Such a
  lambda is read where it is written (or, written inside the body of a
  named lambda that runs at its calls, one that a call names and that is
  not called inside another, with that body at each of that lambda's
  calls), so none of its parameters or reference init captures is linked.
  Each parameter is defined from each call's argument at that call, so a
  write through a pointer or reference parameter reaches no object of the
  caller, and a read through any parameter, a value parameter included,
  misses the argument of a call written after the body, unless the
  definition reaches the read another way: a for, while, or do loop
  holding both the body and the call carries it around its back edge, a
  call of the lambda inside its own body defines the parameter for the
  reads after it, and, for one written inside the body of a named lambda
  that runs at its calls, the definition made in one run of that body
  reaches the reads of a later run (at a later call of that lambda, or
  around the back edge of a loop holding its call); each reference init
  capture holds no value, so a write through it reaches no object of the
  caller and a read through it misses the values of its initializer;
- a write through a reference structured-binding name whose initializer
  or range has no base (`auto& [x, y] = get(p);`, a call): each name is
  then defined from that value, never linked to an object;
- a value passed through an exception: a catch parameter is declared with
  no value, so a thrown value never reaches it;
- a pointer re-pointed between an input and a buffer a device read fills:
  an alias link holds over the whole function, so the device read clears
  the input wherever the re-point lies;
- a device read into one part of an object, which clears the whole
  object, input parts included (field-insensitive): into a member
  (`s.out`), through a name linked with a part (a reference to a member, a
  reference structured-binding name, a reference init capture, or a named
  lambda's linked parameter), or inside a member function that names any
  data member;
- output channels outside the sink list (fprintf, fputc, `<<`, filesystem
  copies, mmap, system), which read as not checked when no listed output
  write is present;
- a stream's write or read whose base is not a variable or a parameter
  declared with a listed stream type: a stream declared with auto, a
  temporary, a data member (`file` in a method, `s.file`), or another
  stream type (basic_ofstream<char>);
- input channels outside the source list (`>>`, getline, mmap);
- a model's own wrapper named `lassi_io_*`, whose definition is never read.

False positives, which read True, each costing the attempt R = -1:
- one struct or array that holds input and output data (field-insensitive);
- a member of an object that holds input data, other than a METADATA
  member (A.rows, or A.values in A.values.size(), unlike a.dims), and a
  structured-binding name read as all of that object's, that
  initializer's, or that range's input, so a shape scalar so taken and
  passed to a library call on the device result (untilize_nfaces) or
  copied into the output object, or a scalar member used in the
  conversion, reads True;
- a pointer re-pointed from an input buffer to an output buffer (alias
  classes never split);
- an output X initialized from input data and then overwritten by
  anything but a whole-variable assignment `X = R` to X itself or a device
  read into X or into a name linked with it (element writes, bulk copies
  such as memcpy or std::copy, assign, swap, and an assignment through a
  name linked with X, such as a reference, a reference structured-binding
  name, or a reference init capture, are weak updates);
- a buffer refilled through a read API outside DEVICE_READS;
- a device read whose destination is named through a macro
  (`#define RESULT res`), which kills nothing, so input staged earlier in
  that buffer reads True;
- a device read through a name with no alias link to the buffer it
  refers to, which kills only that name, so input staged earlier in that
  buffer reads True: a named lambda's pointer or reference parameter whose
  calls do not all pass it one same base, a pointer or reference
  parameter or a reference init capture of a named lambda called inside
  another, a pointer or reference bound to a call or to a conditional, a
  reference structured-binding name whose initializer or range has no
  base (a call) or is a conditional, and a range-for reference element
  over a call (see the misses for each), and the parameter of a lambda
  that no rule binds, such as one passed to std::ranges::for_each or
  std::invoke, or a lambda variable passed to std::for_each;
- a staging write after the read-back inside one loop, when the same
  buffer feeds the output and no device read refills it inside that loop
  (a later definition in a loop reaches an earlier read in it through the
  back edge);
- every arm of a conditional other than `#if 0` and `#if false` is read
  as code, so a CPU fallback under `#else`, `#ifdef`, or `#ifndef` that
  writes the output from the inputs reads True beside the device path;
- input values echoed or logged through a listed write: every listed
  write is an output write wherever it writes (stdout, stderr, a log
  file);
- a main in a model file the build never compiles (a .c file, or a header
  nothing includes): every free function named main outside a namespace
  in the host text is checked beside the program's main;
- file-scope variables of one name are one symbol across files and
  namespaces (file-static and namespaced ones included), so a same-named
  input staging variable and output variable read as one;
- an own constructor passes every argument into its object (member
  initializers are not read), so an input-derived count given to one
  makes the whole object carry input;
- in a member function of a class whose base clause names anything outside
  the host text, an undeclared name that is no macro of the host text
  (FLT_MAX or M_PI from a system header) reads as a data member;
- a lambda that no call in its function names (one passed to a call such
  as std::invoke or a helper, or kept in a container), and a named lambda
  called inside another (see the misses), are read where they are
  written (or, written inside the body of a named lambda that runs at its
  calls, with that body at each of that lambda's calls), so a conversion
  lambda read between the staging and the read-back reads the staged
  input through a capture, also when it is called inside a named lambda
  that does the read-back first, and a device-read lambda read before the
  staging leaves the input staged later in the buffer it refills;
- overloads and same-named methods united by name and arity;
- misparses (template arguments with commas inside call arguments or
  other expressions, digraphs, unusual declarators), which can also miss;
- kernels created only through an API outside KERNEL_CREATORS, such as,
  possibly, LightMetalReplay (a class at
  experimental/lightmetal/lightmetal_replay.hpp:20, its constructor at
  :23; whether it creates kernels was not read,
  plans/spikes/p4-guard-identifiers.md), which read as no kernel (rule K).

No value in this module is a measurement.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath

from lassi.core.capabilities import GUARD_STAGE, HostComputeReading
from lassi.core.record import Diagnostic
from lassi.toolchains._cxx_flow import Site, analyze
from lassi.toolchains._cxx_scan import IDENT, Budget, Lexed, Token, Unreadable, lex, parse_program, resolve_include
from lassi.toolchains.ttmetal_build import is_kernel_source

# The identifier sets, facts of tt-metal 5280a9cf (plans/spikes/p4-guard-identifiers.md: rx 20261004-162204-exec-29e2,
# rx 20261004-162216-exec-362c, and rx 20261004-162228-exec-16ba).
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
# The suffixes of the model's files that are host text, matched as PurePosixPath(path).suffix.
HOST_SUFFIXES = frozenset({".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".inc", ".inl", ".ipp"})
# More bytes of UTF-8 than this, summed over every file read, make the host text unreadable.
SIZE_LIMIT = 4194304
# The reader's step budget (lassi.toolchains._cxx_scan Budget): a reading that needs more steps is not checked.
STEP_LIMIT = 5_000_000

TAG_CODE = "guard-data-movement-only"
HOST_COMPUTE_CODE = "guard-host-compute"
NOT_CHECKED_CODE = "guard-not-checked"

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


@dataclass(frozen=True)
class HostFile:
    """One file of the host text: its path, its lexing, and whether it is one of the model's files."""

    path: str
    lexed: Lexed
    model: bool


def read_host_compute(files: Mapping[str, str], harness: Mapping[str, str]) -> HostComputeReading:
    """Return the CPU -> TT guard's reading of the program built from `files` with `harness` (module docstring).

    `files` are the model's files and `harness` the item's support files,
    each build-directory path -> text; neither is changed. Every input gives
    a reading: a defect in the reader gives None with a note naming the
    exception's class.
    """
    try:
        return _reading(files, harness)
    except Exception as error:  # The reading (module docstring): a reader defect reads as not checked, run goes on
        return HostComputeReading(None, (_unreadable(f"the reader failed with {type(error).__name__}"),))


def _reading(files: Mapping[str, str], harness: Mapping[str, str]) -> HostComputeReading:
    """Return the reading by the steps of the module docstring's The reading."""
    try:
        host = host_text(files, harness)
        analysis = analyze(parse_program([item.lexed for item in host], Budget(STEP_LIMIT)), DEVICE_READS)
    except Unreadable as error:
        return HostComputeReading(None, (_unreadable(error.reason),))
    creates_kernel, creates_compute, creator = _names(host)
    tag = [] if not creates_kernel or creates_compute or creator is None else [_tag(creator)]
    if not analysis.sinks:
        return HostComputeReading(None, (*tag, _note(NOT_CHECKED_CODE, NO_OUTPUT_MESSAGE)))
    if not creates_kernel:
        first = analysis.sinks[0]
        return HostComputeReading(True, (_note(HOST_COMPUTE_CODE, NO_KERNEL_MESSAGE, first.path, first.line),))
    if analysis.violation is not None:
        source, sink = analysis.violation
        message = (
            f"CPU -> TT guard: values read from the inputs at {_site_text(source)} reach the output written at "
            f"{_site_text(sink)} through host code, not through a device read; host_compute is set (Harness Contract)"
        )
        return HostComputeReading(True, (*tag, _note(HOST_COMPUTE_CODE, message, sink.path, sink.line)))
    return HostComputeReading(False, tuple(tag))


def host_text(files: Mapping[str, str], harness: Mapping[str, str]) -> list[HostFile]:
    """Return the host text of a build, lexed, in sorted path order; Unreadable when it cannot be read.

    The roots and the include closure are the module docstring's Host text:
    the roots are read in sorted path order, then the files they include,
    breadth-first, and the first file that cannot be read names the reason.
    """
    roots = sorted(path for path in files if PurePosixPath(path).suffix in HOST_SUFFIXES and not is_kernel_source(path))
    queue, seen = deque(roots), set(roots)
    read: dict[str, HostFile] = {}
    total = 0
    while queue:
        path = queue.popleft()
        model = path in files
        text = files[path] if model else harness[path]
        total += len(text.encode("utf-8", "surrogatepass"))
        if total > SIZE_LIMIT:
            raise Unreadable(f"more than {SIZE_LIMIT} bytes of host text")
        lexed = lex(text, path)
        read[path] = HostFile(path, lexed, model)
        for name, quoted in lexed.includes:
            target = resolve_include(path, name, quoted, files, harness)
            if target is not None and target not in seen:
                seen.add(target)
                queue.append(target)
    return [read[path] for path in sorted(read)]


def _names(host: Sequence[HostFile]) -> tuple[bool, bool, tuple[str, Token] | None]:
    """Return creates_kernel, creates_compute, and the (path, token) the tag sits at (None with no creator)."""
    creates_kernel = creates_compute = False
    in_model: tuple[str, Token] | None = None
    anywhere: tuple[str, Token] | None = None
    for item in host:
        first: Token | None = None
        for token in _named_tokens(item.lexed):
            if token.text in COMPUTE_CONFIGS:
                creates_compute = True
            elif token.text in KERNEL_CREATORS:
                creates_kernel = True
                first = token if first is None or token.pos < first.pos else first
        if first is not None:
            anywhere = anywhere or (item.path, first)
            if item.model:
                in_model = in_model or (item.path, first)
    return creates_kernel, creates_compute, in_model or anywhere


def _named_tokens(lexed: Lexed) -> Iterable[Token]:
    """Yield the identifier tokens of a file's code and of its directive lines other than `#include` lines."""
    for token in lexed.tokens:
        if token.kind == IDENT:
            yield token
    for directive in lexed.directives:
        if len(directive.tokens) > 1 and directive.tokens[1].text == "include":
            continue  # a header name is no identifier (module docstring, The rules)
        for token in directive.tokens:
            if token.kind == IDENT:
                yield token


def _tag(creator: tuple[str, Token]) -> Diagnostic:
    """Return the data-movement-only tag at the kernel-creator token `creator`."""
    path, token = creator
    return _note(TAG_CODE, TAG_MESSAGE, path, token.line, severity="warning")


def _unreadable(reason: str) -> Diagnostic:
    """Return the not-checked note of host text that could not be read for `reason`."""
    message = (
        f"CPU -> TT guard: the host code could not be read ({reason}), so host compute was not checked "
        "(Harness Contract)"
    )
    return _note(NOT_CHECKED_CODE, message)


def _note(
    code: str, message: str, file: str | None = None, line: int | None = None, severity: str = "note"
) -> Diagnostic:
    """Return a parse-stage guard Diagnostic; the message is made plain ASCII (_ascii)."""
    return Diagnostic(stage=GUARD_STAGE, severity=severity, code=code, file=file, line=line, message=_ascii(message))


def _ascii(text: str) -> str:
    """Return `text` with each character outside printable ASCII written as a backslash escape."""
    out = []
    for char in text:
        if " " <= char <= "~":
            out.append(char)
        elif ord(char) < 128:
            out.append(f"\\x{ord(char):02x}")
        else:
            out.append(char.encode("ascii", "backslashreplace").decode("ascii"))
    return "".join(out)


def _site_text(site: Site) -> str:
    """Return a Site as <file>:<line>."""
    return f"{site.path}:{site.line}"
