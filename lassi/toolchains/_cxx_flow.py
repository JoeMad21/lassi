"""Generic data-flow analysis over C and C++ host text (task P4.12).

analyze() answers one question about a program that lassi.toolchains.
_cxx_scan has read: do values read from the program's inputs reach an
output write through host code, without passing through a device read? The
caller names the device-read calls (`device_reads`), so this module names
nothing from any device library; lassi.toolchains.ttmetal_guard gives it
tt-metal's. Nothing here opens a file or starts a process, and every pass is
an iteration over an explicit worklist; the one recursion, which follows an
auto local's initializer to decide whether an argument is a count (Counts
below), stops at depth 8.

Positions. Each event of a function sits at a position: its token's index
in the file, then a sub-position 0 for reads, 1 for kills, and 2 for
definitions. Positions are compared only within one function. A named
lambda is a variable initialized with `=` and exactly one lambda. A call
names it when the call is no member call and its callee is unqualified and
resolves to it. Such a call is that lambda's call only: a local name hides
every function of its name, so no rule for a named function or a library
call (Values, Events, Own functions) applies to it, whatever its name
(fill, read, write, or an own function's), and as a value it contributes
its arguments' sources. A named lambda that a call in the function names
(`auto f = [&]() {...}; ... f();`) is enclosed when its body lies inside
the body of another such lambda, or any of its calls lies inside the body
of such a lambda (its own included). A named lambda that a call names and
that is not enclosed runs where it is called: at each call its unlinked
parameters are killed and defined from the call's arguments, then its
body's events follow in their own order, before the events of any later
token. A pointer- or reference-declared parameter of it is linked with
base(a) instead (Visibility) when every call of the lambda in the function
passes it an argument a and all those arguments have one same base;
otherwise it is unlinked, so a write through it does not reach an argument
and a device read through it kills only the parameter. Every other lambda
stays where it is written (or, written inside the body of a lambda that
runs where it is called, moves with that body's events to each call of
that lambda): one that is no named lambda, one that no call names, and an
enclosed one. Since an enclosed lambda's body is never read at its own
calls (it stays where it is written, or, written inside the body of a
lambda that runs where it is called, moves with that body, as above), all
its parameters are unlinked (killed and defined at each call as above)
and its reference init captures are linked with nothing (Value edges).

Values. sources(E), what an expression E contributes as a value, is
mentions(E) (lassi.toolchains._cxx_scan; a qualified name of a file-scope
variable counts) read with these rules: a device read, a size call, and
the receiver of an own method call contribute nothing directly; an own call
contributes the sources its summary returns to RET (Own functions); a
standard container built with parentheses contributes its value arguments
only (Counts); a lambda contributes its body's sources; an inner
assignment `L op= R` contributes L's mentions and sources(R) [a choice:
the outer value takes R's value, so `x = y = v` gives x the value of v, and
a designated initializer `.m = v` gives v]; and every other call
contributes its receiver's and its arguments' sources.

Events. A call of a named lambda gives only the events Positions states,
none of those below.

- Sources, INPUT definitions: `lassi_io_read(p, X)` (2 arguments) into
  base(X); `fread(p, s, n, f)` (4) into base(p); `X.read(p, n)` into
  base(p) when base(X) is a variable or a parameter whose declared type
  names ifstream, fstream, or istream; POSIX `read(fd, p, n)` (3, not a
  member call) into base(p).
- Sinks: `lassi_io_write(...)` with exactly 6 arguments reads the sources
  of the last; `fwrite(p, s, n, f)` (4) reads sources(p), and so does
  `X.write(p, n)` when base(X) is a variable or a parameter whose declared
  type names ofstream, fstream, or ostream; POSIX `write(fd, p, n)` (3,
  not a member call) reads sources(p). Only base(X)'s own declared type
  names are read, so a stream declared with auto, a temporary, a data
  member (THIS, or the object it belongs to, is the base), and another
  stream type (basic_ofstream<char>) give no event. Other arities are not
  events. fread and fwrite may be std-qualified; POSIX read and write may
  be global-qualified only. A `lassi_io_` callee, whatever its qualifiers,
  is never an own call; fread, fwrite, read, write, and a stream's read and
  write are events only when no own definition matches. Every sink counts
  wherever it writes (stdout, stderr, or any file).
- Device reads: a callee whose last name (an unqualified free callee's
  macro alias expanded one level) is one of `device_reads` is never an own
  call, and kills base(a) of each argument a that is an operand (one name
  with only member accesses, subscripts, and calls, after base()'s
  stripping: `res`, `c.data()`, `&c[0]`, `*p`; not `af != nullptr` or
  `n * 4`). Arguments are never macro-expanded: with `#define RESULT res`,
  the argument RESULT kills only the undeclared name RESULT, never res.
- Value edges: an assignment `L op= R` defines base(L) from sources(R) at
  its operator (`operator=`, `= default`, `= delete`, an init capture in a
  lambda introducer, and a left side with no base, such as a designated
  initializer, define nothing). `L = R` with L a single value-declared
  identifier other than THIS, or a name of a reference structured binding
  defined as a value binding's (below), first kills L's class unless
  sources(R) reads that class (a strong update; R's own calls read what
  their summaries return to RET, so the decision is taken again once
  summaries exist, until it no longer changes); an identifier with no
  declaration counts as value-declared [a choice: a kill can only remove
  flows]. `L = R` with L a single pointer-declared identifier links L with
  base(R) instead of defining anything. A value- or array-declared name's
  initializer defines it at the token after the name; a pointer- or
  reference-declared name's initializer links it with base(E) (no link
  when there is no base). A range-for over E links a reference-declared
  element with base(E) and otherwise defines it from sources(E) at the
  `:`. The names of a structured binding follow the same rules: a
  reference binding (`auto& [x, y] = E`, `for (const auto& [x, y] : E)`)
  links each name with base(E) and a value binding defines each from
  sources(E), and a reference binding of an E with no base (a call)
  defines each from sources(E) as a value binding does. A reference init
  capture `[&x = E]` links x with base(E), unless its lambda is enclosed
  (Positions). A catch parameter, a value init capture's name, and an
  enclosed lambda's reference init capture are declared with no
  initializer, so they start with no value: neither the thrown value nor
  the capture's initializer is read into them. `return E` outside a lambda
  body defines RET at the `return`.
- Counts carry no value. A call of size, ssize, or distance (std- or
  `::`-qualified or unqualified; not a member call, and not an own call)
  contributes nothing. A standard sequence container (CONTAINERS: vector,
  deque, list, forward_list, valarray, basic_string, string) built with
  parentheses, as a declaration `T X(args)` whose head is one or as a
  temporary `T(args)` that is not an own call, and `X.assign(args)`, read
  their arguments so: when the first argument is a count, `(n)` reads
  nothing and `(n, v, ...)` reads only v; otherwise every argument is a
  value (`(first, last)`, `(other)`). `X.insert(pos, n, v)` reads only v
  when n is a count. An argument is a count when it is no pointer or
  iterator producer and its first mention names something that holds no
  elements: not pointer- or array-declared, not a container or a view by
  its declared type names (for a data member, only its types from the
  class count), and, for an auto local, not initialized by a producer, by
  a container or view temporary, or by the name, or std::move of the name,
  of such a local (followed to depth 8). An argument with no mention
  (`1024`) is no count and reads nothing either way. Braces are always
  values.
- Metadata never links, defines, or kills: base() of `X.m` with m in
  METADATA is no base here (`const uint64_t* shape = a.dims;` links
  nothing), and `X.m` is no mention of X.
- Standard calls: std-qualified (`std::`, `::std::`), or unqualified or
  `::`-qualified with no own definition that admits the call; an
  execution-policy first argument (`std::execution::...`,
  `::std::execution::...`, or `execution::...`) is dropped first. Each row
  defines base(target) from the sources of the value arguments at the
  call (n is the argument count):
  memcpy, memmove, memset (n = 3): arg 0 from arg 1.
  copy, move, copy_backward, move_backward (n = 3): arg 2 from args 0, 1.
  copy_if, remove_copy, remove_copy_if, replace_copy, replace_copy_if,
  unique_copy, reverse_copy, partial_sum, adjacent_difference,
  inclusive_scan, exclusive_scan, transform_inclusive_scan,
  transform_exclusive_scan (n >= 3): arg 2 from every other argument.
  copy_n (n = 3): arg 2 from args 0 and 1.
  transform (n = 4): arg 2 from args 0, 1, 3; transform (n = 5): arg 3
  from args 0, 1, 2, 4.
  fill, fill_n, generate, generate_n (n >= 2): arg 0 from the last arg.
  iota (n = 3): arg 0 from arg 2.
  swap (n = 2): each from the other.
  for_each, for_each_n (n >= 2): when the last argument is a lambda, its
  first parameter is linked with base(arg 0) when reference-declared and
  otherwise defined from sources(arg 0).
  The ranges rows take callees qualified `std::ranges::`,
  `::std::ranges::`, or `ranges::` (the last only when no own definition
  admits the call), and drop no policy:
  ranges::copy (n = 2): arg 1 from arg 0. ranges::transform (n = 3): arg 1
  from args 0 and 2; ranges::transform (n = 4): arg 2 from args 0, 1, 3.
- Member calls on X that are not own calls define base(X): assign (Counts
  above), insert (3 arguments: Counts above; otherwise every argument but
  the first), emplace (every argument but the first), push_back,
  emplace_back, push_front, emplace_front, append (every argument),
  resize(n, v) and fill(v) (v), and swap(Y) (Y, and X into base(Y)). No
  other library call defines anything: accumulate, max, an unknown call,
  and a `lassi_io_` call other than the source and sink forms contribute
  only to a value (Values above).

Own functions. Every definition not named `lassi_io_*` is read. A call
matches every definition whose name is the callee's last name (an
unqualified free callee's macro alias expanded one level) and whose
parameters admit its argument count (overloads, same-named methods, and
free functions of the name are united), except that a call of a named
lambda (Positions) and std-qualified, device-read, and `lassi_io_` callees
never match, and a member call `X.f(...)` matches only when X can be an
object of a class in the host text: the declared type names of X's base
(template arguments included; for a data member, or `this->m`, the
member's types from its class) include such a class, or are not known
(`this` itself, auto, no declaration, an unrecorded member, or a receiver
too long to read). An own call wins over the source, sink, standard, and
member rules. Ports are the parameters P0..Pn, THIS (member functions),
the file-scope variables the function or its callees touch, RET, INPUT,
and SINK; the out-ports that can carry values back are the pointer- or
reference-declared parameters (array and view parameters included), THIS,
and file-scope variables. At a call, Pi binds to argument i (sources:
sources(arg i); actual: base(arg i), so an argument with no base, such as
a call or a view temporary `std::span<float>(c)`, has no actual and
nothing comes back through Pi), THIS to the receiver of `X.f(...)`
(unbound when the receiver is too long to read) or, for an unqualified
call inside a member function that does not match only constructors, to
the caller's THIS, and a file-scope variable to itself. The receiver of an
own method call is read only through THIS: it reaches a value only when
the summary returns THIS. A constructor (a member named as its class) also
passes every argument to RET and to THIS, since its member initializers
are not read: a temporary `T(args)` returns its arguments, and a
declaration `T x(args)`, `T x{args}`, or `T x;` of a value-declared x
whose head T is a class of the host text applies the constructors of T
that admit the argument count, with THIS bound to x. Summaries are two
monotone fixpoints over a worklist: kills first (a function kills an
out-port when it holds a kill of the port's class: a device read, a strong
update, or a callee's kill), then flows with kills fixed: X -> SINK when a
sink reads a state reached from X, X -> RET when RET is defined from one,
X -> Y when Y's class is defined from one with no later kill of that class
in the function, and INPUT -> SINK when the function holds a violation.
One pass over the function's events reaches every port at once (each state
carries the set of ports that reach it). Applying a summary at a call q
kills the actual of each killed port at (q, 1), defines the actual of Y
from sources(X) at (q, 2), reads sources(X) as a sink at (q, 0) for
X -> SINK, defines INPUT into the actual of Y for INPUT -> Y, and
contributes sources(X) to a value for X -> RET. INPUT -> SINK is a
violation at the call itself.

Visibility. Alias classes are a union-find over each function's symbols,
linked by pointer and reference declarations (reference structured
bindings, and the reference init captures of lambdas not enclosed,
included), pointer re-points, reference range-fors, for_each reference
parameters, and the named lambdas' parameters that Positions links; links
are never cut, and a link holds at every position of the function, before
the linking statement too. Definitions and kills act on classes. A state
is a class and the position of the definition that gave it a value
(ENTRY, before every event, for a port). A definition at p is visible at
a read r after it when no kill of its class lies between them. It is
visible at a read r before it only around a loop's back edge: r and p
lie in one for, while, or do statement L, no kill of the class lies
between p and L's end, and none between L's head and r (the innermost
loop holding both decides). A loop made with goto is no loop here, so its
back edge carries nothing. The reads of file-scope initializers see every
definition. A violation is a state reached from an INPUT definition that a
sink reads visibly, a sink whose value holds an own call that returns
INPUT, or a call of a function that has INPUT -> SINK. The graphs checked
are those of every main (a free function named main outside any namespace)
in the host text, a second main in a file the build never compiles
included, each with its applied summaries and the definitions of value- or
array-declared file-scope initializers from the file-scope variables they
mention; with no main, every function's graph is checked, without them.
The witness is the violation with the least file rank of the function
checked, then sink position, then input token; an applied summary's events
sit at its call site, so a violation inside a callee is reported at the
call. Only the witness's input and output positions are reported, not the
path between them, and only functions that some function calls get
summaries (a summary is read only at a call).

Cost and budget. The intended costs, never measured (PROJECTED): the
extraction reads each function's tokens a bounded number of times, apart
from the scans that lassi.toolchains._cxx_scan bounds (Bounds there); the
label reach of each main fires each definition once; the summary reach
fires each definition once per port that reaches it, so one summary step
costs about (definitions + states) x ports; a function is stepped again
only when a callee's summary grew; and both fixpoints are run again for the
strong-update decisions until they hold, at most once more than the number
of strong updates whose right side calls an own function. Every pass that
can repeat work spends the program's Budget (lassi.toolchains._cxx_scan),
so a reading that needs more steps ends with Unreadable ("the reader
exceeded its budget of <n> steps") rather than run long. No value in this
module is a measurement.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict, deque
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace

from lassi.toolchains._cxx_scan import (
    ARRAY,
    ASSIGN_OPS,
    FREE,
    GLOBAL,
    POINTER,
    PUNCT,
    REFERENCE,
    UNEVALUATED,
    VALUE,
    VIEW_TYPES,
    Body,
    Budget,
    Callee,
    Decl,
    Function,
    Program,
    RangeFor,
    Return,
    walk_body,
)

# The prefix of the harness I/O functions, whose definitions are never summarized.
HARNESS_PREFIX = "lassi_io_"
# Ports: parameters are ("param", i) and file-scope variables ("global", symbol).
THIS_PORT = ("this", 0)
RET_PORT = ("ret", 0)
INPUT_PORT = ("input", 0)
SINK_PORT = ("sink", 0)
# How a call is read.
LAMBDA, HARNESS, DEVICE, OWN, LIBRARY = "lambda", "harness", "device", "own", "library"
# The declared stream types whose read is a source and whose write is a sink.
IN_STREAMS = frozenset({"ifstream", "fstream", "istream"})
OUT_STREAMS = frozenset({"ofstream", "fstream", "ostream"})
# The standard calls of the module docstring's table, by the rule each follows.
COPIES = frozenset({"copy", "move", "copy_backward", "move_backward"})
COPY_FAMILY = frozenset(
    {"copy_if", "remove_copy", "remove_copy_if", "replace_copy", "replace_copy_if", "unique_copy", "reverse_copy"}
)
FILLS = frozenset({"fill", "fill_n", "generate", "generate_n"})
SCANS = frozenset(
    {"partial_sum", "adjacent_difference", "inclusive_scan", "exclusive_scan", "transform_inclusive_scan",
     "transform_exclusive_scan"}
)
APPENDS = frozenset({"push_back", "emplace_back", "push_front", "emplace_front", "append"})
# The calls whose result is a size or a distance, never an element value.
SIZE_CALLS = frozenset({"size", "ssize", "distance"})
# The standard sequence containers, whose single parenthesized count sizes them and carries no value.
CONTAINERS = frozenset({"vector", "deque", "list", "forward_list", "valarray", "basic_string", "string"})
# The qualifiers of a standard call and of a ranges call ("" is a leading global `::`).
STD_QUALIFIERS = ((), ("",), ("std",), ("", "std"))
RANGES_QUALIFIERS = (("std", "ranges"), ("ranges",), ("", "std", "ranges"))
# A position before every event of a function, and one after every event.
ENTRY = -1
END = 1 << 62

Port = tuple[str, int]
State = tuple[int, int]


@dataclass(eq=False)
class Src:
    """What an expression contributes as values: symbols, sub-expressions, and own calls (whose RET flows count)."""

    symbols: set[int] = field(default_factory=set)
    parts: list[Src] = field(default_factory=list)
    calls: list[Call] = field(default_factory=list)


@dataclass(eq=False)
class Call:
    """One call of own functions at token `at`: the definitions it matches and its bindings.

    `args[i]` is sources(Pi) and `bases[i]` actual(Pi); `this_src` and
    `this_base` are THIS's (None when unbound); `pos` is the call's base
    position once placed (Positions in the module docstring).
    """

    at: int
    callees: tuple[int, ...]
    args: list[Src]
    bases: list[int | None]
    this_src: Src | None = None
    this_base: int | None = None
    pos: int = 0


@dataclass
class Facts:
    """One function's events, placed at base positions (token index x width + offset), before summaries apply.

    `defs` hold (position, token, symbol, source), `kills` (position,
    symbol), `strong` the strong-update candidates (position, symbol,
    right side), `inputs` (position, token, symbol), `sinks` (position,
    token, source), and `loops` the base-position ranges of loop statements.
    """

    index: int
    body: Body
    defs: list[tuple[int, int, int, Src]] = field(default_factory=list)
    kills: list[tuple[int, int]] = field(default_factory=list)
    strong: list[tuple[int, int, Src]] = field(default_factory=list)
    links: list[tuple[int, int]] = field(default_factory=list)
    inputs: list[tuple[int, int, int]] = field(default_factory=list)
    sinks: list[tuple[int, int, Src]] = field(default_factory=list)
    calls: list[Call] = field(default_factory=list)
    loops: list[tuple[int, int]] = field(default_factory=list)
    globals_used: set[int] = field(default_factory=set)
    parent: dict[int, int] = field(default_factory=dict)

    def find(self, symbol: int) -> int:
        """Return the alias class (union-find root) of `symbol`."""
        parent = self.parent
        root = symbol
        while parent.get(root, root) != root:
            root = parent[root]
        while parent.get(symbol, symbol) != root:
            parent[symbol], symbol = root, parent[symbol]
        return root

    def union(self, a: int, b: int) -> None:
        """Put `a` and `b` in one alias class."""
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


@dataclass(frozen=True)
class Site:
    """A place in the host text: file rank, token index, path, and 1-based line."""

    rank: int
    index: int
    path: str
    line: int


@dataclass(frozen=True)
class Analysis:
    """What analyze() found: every output write in (rank, position) order, and the witness (source, sink) or None."""

    sinks: tuple[Site, ...]
    violation: tuple[Site, Site] | None


def analyze(program: Program, device_reads: Collection[str]) -> Analysis:
    """Return the output writes and the input-to-output witness of `program` (see the module docstring).

    The work spends `program.budget` when there is one; Unreadable is raised
    when it runs out.
    """
    return _Analyzer(program, frozenset(device_reads)).run()


def _spend(budget: Budget | None, steps: int) -> None:
    """Spend `steps` of `budget`, when there is one."""
    if budget is not None:
        budget.spend(steps)


class _Extractor:
    """Reads one function's events from its walked body."""

    def __init__(self, program: Program, index: int, own: dict[str, list[int]], device_reads: frozenset[str]) -> None:
        """Walk the body and prepare the per-function caches."""
        self.program = program
        self.budget = program.budget
        self.body = walk_body(program, index)
        self.code = self.body.code
        self.facts = Facts(index, self.body)
        self.own = own
        self.device_reads = device_reads
        self.memo: dict[tuple[int, int], Src] = {}
        self.pending: list[tuple[int, int, Src]] = []
        self.kinds: dict[int, str] = {}
        self.lambda_callees: dict[int, int] = {}
        self.arguments: dict[int, list[tuple[int, int]]] = {}
        self.sites: dict[int, Call] = {}
        self.matched: dict[tuple[str, int], tuple[int, ...]] = {}
        self.raw_defs: list[tuple[int, int, Src]] = []
        self.raw_kills: list[tuple[int, int]] = []
        self.raw_strong: list[tuple[int, int, Src]] = []
        self.raw_inputs: list[tuple[int, int]] = []
        self.raw_sinks: list[tuple[int, Src]] = []
        self.raw_calls: list[Call] = []
        self.lambda_sites: defaultdict[int, list[int]] = defaultdict(list)
        self.lambda_args: defaultdict[int, list[tuple[int, list[tuple[int, int]]]]] = defaultdict(list)
        self.placed: dict[int, tuple[int, int, list[int]]] = {}
        self.value_bindings: set[int] = set()
        self.auto_inits: dict[int, tuple[int, int]] = {}

    def run(self) -> Facts:
        """Return the function's Facts, with its alias classes settled and its events placed.

        Reference init captures and the named lambdas' parameters are bound
        once every call is known, since an enclosed lambda's get no link
        (module docstring, Positions). A whole assignment stays a strong
        update only for a value-declared name or a name in value_bindings.
        """
        for item in self.body.items:
            if isinstance(item, Decl) and item.init is not None and not self.program.symbols[item.symbol].types:
                self.auto_inits[item.symbol] = item.init
        captures: list[Decl] = []
        for item in self.body.items:
            if isinstance(item, Decl) and item.capture is not None:
                captures.append(item)
            elif isinstance(item, Decl):
                self._decl(item)
            elif isinstance(item, RangeFor):
                self._range_for(item)
            elif isinstance(item, Return):
                self.raw_defs.append((item.at, self.body.ret, self.src(*item.range)))
            else:
                self._scan(*item.range)
        symbols = self.program.symbols
        self.raw_strong = [
            update for update in self.raw_strong if symbols[update[1]].kind == VALUE or update[1] in self.value_bindings
        ]
        self.placed = self._placed_lambdas()
        lambdas = self.body.named_lambdas
        enclosed = {lambdas[symbol].intro[0] for symbol in self.lambda_sites if symbol not in self.placed}
        for item in captures:
            if item.capture not in enclosed:
                self._decl(item)
        self._bind_lambda_params()
        while self.pending:
            lo, hi, node = self.pending.pop()
            self._fill(lo, hi, node)
        facts = self.facts
        for a, b in facts.links:
            facts.union(a, b)
        _Placer(self).place()
        return facts

    def resolve(self, i: int) -> int:
        """Return the symbol token `i` names, noting a file-scope variable as used.

        A name that ends a qualified name (`ns::x`, `::x`) resolves to the
        file-scope variable of that name.
        """
        _spend(self.budget, 1)
        name = self.code.text(i)
        if self.code.text(i - 1) == "::" and name in self.program.globals:
            symbol = self.program.globals[name]
        else:
            symbol = self.body.resolve(i)
        if self.program.symbols[symbol].role == GLOBAL:
            self.facts.globals_used.add(symbol)
        return symbol

    def base_symbol(self, lo: int, hi: int) -> int | None:
        """Return the symbol of base(E) for the expression [lo, hi), or None (metadata objects have none)."""
        index = self.code.base_index(lo, hi, metadata=True, qualified=self.program.globals)
        return None if index is None else self.resolve(index)

    def src(self, lo: int, hi: int) -> Src:
        """Return the Src of the expression [lo, hi); it is filled before the extraction ends (memoized by range)."""
        node = self.memo.get((lo, hi))
        if node is None:
            node = Src()
            self.memo[(lo, hi)] = node
            self.pending.append((lo, hi, node))
        return node

    def _fill(self, lo: int, hi: int, node: Src) -> None:
        """Fill `node` with what [lo, hi) contributes; groups and assignments' right sides become parts.

        The receiver of an own method call is skipped: it reaches the value
        only through the call's THIS (Own functions in the module docstring).
        """
        code = self.code
        _spend(self.budget, hi - lo + 1)
        skips = self._own_receivers(lo, hi)
        k, i = 0, lo
        while i < hi:
            while k < len(skips) and skips[k][1] <= i:
                k += 1
            if k < len(skips) and skips[k][0] <= i:
                i = skips[k][1]
                continue
            token = code.tokens[i]
            if token.kind == PUNCT and token.text in ("(", "{"):
                i = self._fill_group(i, node)
                continue
            if token.kind == PUNCT and token.text in ASSIGN_OPS:
                # An inner assignment's value is its left side after the assignment: its right side is read here as
                # a part (the same node its own definition reads), as is a designated initializer's (`.m = v`).
                end = min(code.next_separator(i), hi)
                node.parts.append(self.src(i + 1, end))
                i = max(end, i + 1)
                continue
            skip = code.skip_excluded(i)
            if skip is not None:
                i = max(skip, i + 1)
                continue
            if code.is_mention(i, qualified=self.program.globals):
                node.symbols.add(self.resolve(i))
            i += 1

    def _own_receivers(self, lo: int, hi: int) -> list[tuple[int, int]]:
        """Return the merged token ranges of the receivers of own method calls at the top level of [lo, hi)."""
        code = self.code
        found: list[tuple[int, int]] = []
        i = lo
        while i < hi:
            if code.is_punct(i, ("(", "[", "{")):
                if code.text(i) == "(":
                    callee = code.callee_at(i)
                    if callee is not None and callee.receiver is not None and callee.receiver[0] >= lo:
                        if self.classify(callee) == OWN:
                            found.append(callee.receiver)
                i = code.close(i) + 1
                continue
            i += 1
        merged: list[tuple[int, int]] = []
        for start, end in sorted(found):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        return merged

    def _fill_group(self, i: int, node: Src) -> int:
        """Add the group opening at `i` to `node` (see the module docstring for calls); return its end."""
        code = self.code
        closer = code.close(i)
        if code.text(i) == "(":
            callee = code.callee_at(i)
            if callee is not None:
                kind = self.classify(callee)
                if kind == DEVICE:
                    return closer + 1
                if kind == OWN:
                    node.calls.append(self.call_site(callee))
                    return closer + 1
                if kind == LIBRARY and not callee.member and callee.qualifiers in STD_QUALIFIERS:
                    name = self.callee_name(callee)
                    if name in SIZE_CALLS:
                        return closer + 1
                    if name in CONTAINERS:
                        node.parts.extend(self.src(a, b) for a, b in self._container_values(self.args(callee)))
                        return closer + 1
        node.parts.append(self.src(i + 1, min(closer, code.n)))
        return closer + 1

    def callee_name(self, callee: Callee) -> str:
        """Return the callee's last name, an unqualified free callee's alias expanded one level."""
        if not callee.member and not callee.qualifiers:
            return self.program.aliases.get(callee.name, callee.name)
        return callee.name

    def args(self, callee: Callee) -> list[tuple[int, int]]:
        """Return the argument ranges of a call (memoized)."""
        found = self.arguments.get(callee.open)
        if found is None:
            found = self.code.split_commas(callee.open + 1, min(callee.close, self.code.n))
            self.arguments[callee.open] = found
        return found

    def matches(self, name: str, count: int) -> tuple[int, ...]:
        """Return the own definitions named `name` that admit `count` arguments (memoized)."""
        key = (name, count)
        found = self.matched.get(key)
        if found is None:
            candidates = self.own.get(name, ())
            _spend(self.budget, len(candidates))
            found = tuple(index for index in candidates if self.program.functions[index].admits(count))
            self.matched[key] = found
        return found

    def classify(self, callee: Callee) -> str:
        """Return how a call is read: LAMBDA, HARNESS, DEVICE, OWN, or LIBRARY (memoized).

        An unqualified free call of a name that a named lambda in scope
        declares is LAMBDA, its symbol kept in lambda_callees: the local name
        hides every function of its name (module docstring, Positions).
        """
        kind = self.kinds.get(callee.open)
        if kind is None:
            name = self.callee_name(callee)
            local = None if callee.member or callee.qualifiers else self.body.lookup(callee.index)
            if local is not None and local in self.body.named_lambdas:
                kind = LAMBDA
                self.lambda_callees[callee.open] = local
            elif name.startswith(HARNESS_PREFIX):
                kind = HARNESS
            elif name in self.device_reads:
                kind = DEVICE
            elif _std_qualified(callee.qualifiers) or not self.matches(name, len(self.args(callee))):
                kind = LIBRARY
            elif callee.member and not self._receiver_may_be_own(callee):
                kind = LIBRARY
            else:
                kind = OWN
            self.kinds[callee.open] = kind
        return kind

    def _receiver_may_be_own(self, callee: Callee) -> bool:
        """Return True when the receiver of the member call can be an object of a class in the host text."""
        if callee.receiver is None:
            return True
        types = self._receiver_types(*callee.receiver)
        return types is None or bool(types & self.program.classes)

    def _receiver_types(self, lo: int, hi: int) -> frozenset[str] | None:
        """Return the declared type names of the object [lo, hi) names, or None when they are not known."""
        code = self.code
        index = code.base_index(lo, hi, qualified=self.program.globals)
        if index is None:
            return None
        if code.text(index) == "this":
            if index + 2 < hi and code.text(index + 1) == "->" and code.is_name(index + 2):
                return self.body.members.get(code.text(index + 2))
            return None
        symbol = self.resolve(index)
        if symbol == self.body.this:
            return self.body.members.get(code.text(index))
        info = self.program.symbols[symbol]
        return None if info.role == FREE or not info.types else info.types

    def call_site(self, callee: Callee) -> Call:
        """Return the Call of an own call, made and recorded once."""
        site = self.sites.get(callee.open)
        if site is not None:
            return site
        args = self.args(callee)
        callees = self.matches(self.callee_name(callee), len(args))
        site = Call(callee.index, callees, [self.src(a, b) for a, b in args], [self.base_symbol(a, b) for a, b in args])
        if callee.member:
            if callee.receiver is not None:
                site.this_src = self.src(*callee.receiver)
                site.this_base = self.base_symbol(*callee.receiver)
        elif self.body.this is not None and not callee.qualifiers and not all(map(self._constructor, callees)):
            site.this_src, site.this_base = Src({self.body.this}), self.body.this
        self.sites[callee.open] = site
        self.raw_calls.append(site)
        return site

    def _constructor(self, index: int) -> bool:
        """Return True for a constructor: a member function named as its class."""
        function = self.program.functions[index]
        return function.member and function.name == function.class_name

    def _decl(self, item: Decl) -> None:
        """Define a value- or array-declared name from its initializer, or link a pointer or reference.

        A declaration of a class with constructors also applies them
        (_construct). A reference structured binding whose initializer has
        no base is defined from it, as a value binding is (value_bindings).
        """
        self._construct(item)
        if item.init is None:
            return
        if item.kind not in (VALUE, ARRAY):
            target = self.base_symbol(*item.init)
            if target is not None or not item.binding:
                self._link(item.symbol, target)
                return
            self.value_bindings.add(item.symbol)
        if item.head in CONTAINERS and self.code.text(item.at) == "(":
            ranges = self._container_values(self.code.split_commas(*item.init))
            source = Src(parts=[self.src(a, b) for a, b in ranges])
        else:
            source = self.src(*item.init)
        self.raw_defs.append((item.at, item.symbol, source))

    def _range_for(self, item: RangeFor) -> None:
        """Link a reference-declared range-for name with base(E), or define it from sources(E) at the `:`.

        A reference element of a range with no base gets neither; a
        reference structured binding over one is defined, as a value one is.
        """
        target = self.base_symbol(*item.range) if item.kind == REFERENCE else None
        if target is not None:
            self._link(item.symbol, target)
        elif item.kind != REFERENCE or item.binding:
            if item.kind == REFERENCE:
                self.value_bindings.add(item.symbol)
            self.raw_defs.append((item.at, item.symbol, self.src(*item.range)))

    def _construct(self, item: Decl) -> None:
        """Apply the constructors of the declared class at a `T x(args)`, `T x{args}`, or `T x;` declaration."""
        if item.kind != VALUE or item.head not in self.program.classes:
            return
        if item.init is not None and self.code.text(item.at) not in ("(", "{"):
            return
        args = self.code.split_commas(*item.init) if item.init is not None else []
        callees = tuple(index for index in self.matches(item.head, len(args)) if self._constructor(index))
        if not callees:
            return
        site = Call(item.at, callees, [self.src(a, b) for a, b in args], [self.base_symbol(a, b) for a, b in args])
        site.this_src, site.this_base = Src({item.symbol}), item.symbol
        self.raw_calls.append(site)

    def _container_values(self, args: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
        """Return the arguments of a container's parenthesized constructor or assign that are values (counts not)."""
        if not args:
            return []
        if self._is_count(*args[0]):
            return list(args[1:2])
        return list(args)

    def _is_count(self, lo: int, hi: int) -> bool:
        """Return True when the argument [lo, hi) is a count: no producer, and its first mention is no container."""
        if self.code.producer(lo, hi):
            return False
        first = next(self.code.iter_mentions(lo, hi, qualified=self.program.globals), None)
        return first is not None and not self._container_like(self.resolve(first), first, 0)

    def _container_like(self, symbol: int, token: int, depth: int) -> bool:
        """Return True when `symbol` (named at `token`) holds or points into elements (module docstring, Counts)."""
        info = self.program.symbols[symbol]
        if symbol == self.body.this:
            name = self.code.text(token + 2) if self.code.text(token) == "this" else self.code.text(token)
            types = self.body.members.get(name, frozenset())
        elif info.kind in (POINTER, ARRAY):
            return True
        else:
            types = info.types
        if types & CONTAINERS or types & VIEW_TYPES:
            return True
        init = self.auto_inits.get(symbol)
        if types or init is None or depth > 8:
            return False
        return self._container_init(*init, depth)

    def _container_init(self, lo: int, hi: int, depth: int) -> bool:
        """Return True when the auto initializer [lo, hi) yields a container, a view, or a pointer into one."""
        code = self.code
        if code.producer(lo, hi):
            return True
        start = lo + 2 if code.text(lo) == "std" and code.text(lo + 1) == "::" else lo
        if code.text(start) in CONTAINERS or code.text(start) in VIEW_TYPES:
            return True
        if code.text(start) == "move" and code.text(start + 1) == "(":
            lo, hi = start + 2, code.close(start + 1)
        if hi - lo == 1 and code.is_name(lo):
            return self._container_like(self.resolve(lo), lo, depth + 1)
        return False

    def _link(self, a: int, b: int | None) -> None:
        """Link `a` with `b` into one alias class when `b` is a symbol."""
        if b is not None and a != b:
            self.facts.links.append((a, b))

    def _scan(self, lo: int, hi: int) -> None:
        """Read the assignments and calls of the expression [lo, hi); lambda bodies are walked as statements."""
        code = self.code
        _spend(self.budget, hi - lo + 1)
        i = lo
        while i < hi:
            token = code.tokens[i]
            if token.kind == PUNCT:
                if token.text == "[":
                    found = code.lambda_at(i)
                    if found is not None or code.text(i + 1) == "[":
                        i = max((found.body[1] if found is not None else code.close(i)) + 1, i + 1)
                        continue
                elif token.text in ASSIGN_OPS:
                    self._assignment(i, lo, hi)
                elif token.text == "(":
                    callee = code.callee_at(i)
                    if callee is not None:
                        self._call(callee)
            elif token.text in UNEVALUATED:
                i = code.skip_unevaluated(i)
                continue
            i += 1

    def _assignment(self, k: int, lo: int, hi: int) -> None:
        """Read the assignment whose operator is token `k` (see the module docstring, Value edges)."""
        code = self.code
        if code.text(k - 1) == "operator" or code.text(k) == "=" and code.text(k + 1) in ("default", "delete"):
            return
        start = code.segment_start(k, lo)
        if start >= k:
            return
        right = (k + 1, min(code.next_separator(k), hi))
        if k - start == 1 and code.is_name(start) and code.text(k) == "=" and self._whole(start, k, right):
            return
        target = self.base_symbol(start, k)
        if target is not None:
            self.raw_defs.append((k, target, self.src(*right)))

    def _whole(self, name: int, k: int, right: tuple[int, int]) -> bool:
        """Read `X = R` for the single identifier X at `name`; return True when it defines nothing (a re-point).

        A value- or reference-declared X other than THIS is noted as a strong
        update; run keeps only those of value-declared names and of
        value_bindings.
        """
        symbol = self.resolve(name)
        kind = self.program.symbols[symbol].kind
        if kind == POINTER:
            self._link(symbol, self.base_symbol(*right))
            return True
        if kind in (VALUE, REFERENCE) and symbol != self.body.this:
            self.raw_strong.append((k, symbol, self.src(*right)))
        return False

    def _call(self, callee: Callee) -> None:
        """Read the events of one call (see the module docstring)."""
        kind = self.classify(callee)
        if kind == LAMBDA:
            self._lambda_call(self.lambda_callees[callee.open], callee)
            return
        if kind == OWN:
            self.call_site(callee)
            return
        args, name, at = self.args(callee), self.callee_name(callee), callee.index
        if kind == HARNESS:
            if name == "lassi_io_read" and len(args) == 2:
                self._input(args[1], at)
            elif name == "lassi_io_write" and len(args) == 6:
                self._sink(args[5], at)
        elif kind == DEVICE:
            for a, b in args:
                symbol = self.base_symbol(a, b) if self.code.is_operand(a, b) else None
                if symbol is not None:
                    self.raw_kills.append((at, symbol))
        elif callee.member:
            self._member_call(callee, name, args, at)
        else:
            self._free_call(callee.qualifiers, name, args, at)

    def _lambda_call(self, symbol: int, callee: Callee) -> None:
        """Note a call of the named lambda `symbol` and its arguments; _bind_lambda_params binds them."""
        self.lambda_sites[symbol].append(callee.index)
        self.lambda_args[symbol].append((callee.index, self.args(callee)))

    def _placed_lambdas(self) -> dict[int, tuple[int, int, list[int]]]:
        """Return (body start, body end, call tokens) of each named lambda that runs at its calls, by symbol.

        A named lambda that a call names is enclosed, and is never read at its
        own calls (it stays where it is written, or, written inside the body
        of a named lambda that runs at its calls, moves with that body), when
        its body, or any of its calls, lies inside the body of a named lambda
        that a call names (for a call, its own body included).
        """
        candidates = [
            (symbol, self.body.named_lambdas[symbol].body, sites)
            for symbol, sites in self.lambda_sites.items() if symbol in self.body.named_lambdas
        ]
        _spend(self.budget, len(candidates) * len(candidates))
        placed: dict[int, tuple[int, int, list[int]]] = {}
        for symbol, (start, end), sites in candidates:
            enclosed = any(
                (other, other_end) != (start, end) and other <= start <= other_end
                or any(other <= site <= other_end for site in sites)
                for _, (other, other_end), _ in candidates
            )
            if not enclosed:
                placed[symbol] = (start, end, sorted(set(sites)))
        return placed

    def _bind_lambda_params(self) -> None:
        """Bind each named lambda's parameters to the arguments of its calls (module docstring, Positions).

        A pointer- or reference-declared parameter of a lambda that runs at
        its calls is linked with base(argument) when every call passes it an
        argument and all those arguments have one same base; every other
        parameter is killed and defined from its argument at each call that
        passes one.
        """
        for symbol, calls in self.lambda_args.items():
            params = self.body.lambda_params.get(self.body.named_lambdas[symbol].intro[0], ())
            linkable = symbol in self.placed
            for index, param in enumerate(params):
                if param is None:
                    continue
                bound = [(at, args[index]) for at, args in calls if index < len(args)]
                pointer = self.program.symbols[param].kind in (POINTER, REFERENCE)
                if linkable and pointer and len(bound) == len(calls):
                    bases = {self.base_symbol(*argument) for _, argument in bound}
                    if len(bases) == 1 and None not in bases:
                        self._link(param, bases.pop())
                        continue
                for at, (a, b) in bound:
                    self.raw_kills.append((at, param))
                    self.raw_defs.append((at, param, self.src(a, b)))

    def _free_call(self, qualifiers: tuple[str, ...], name: str, args: list[tuple[int, int]], at: int) -> None:
        """Read a call that is not a member call: a source, a sink, or a standard call."""
        standard = qualifiers in STD_QUALIFIERS
        plain = qualifiers in ((), ("",))
        count = len(args)
        if standard and name == "fread" and count == 4:
            self._input(args[0], at)
        elif standard and name == "fwrite" and count == 4:
            self._sink(args[0], at)
        elif plain and name == "read" and count == 3:
            self._input(args[1], at)
        elif plain and name == "write" and count == 3:
            self._sink(args[1], at)
        elif standard:
            self._standard(name, self._policy_dropped(args), at)
        elif qualifiers in RANGES_QUALIFIERS:
            self._ranges(name, args, at)

    def _policy_dropped(self, args: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Return `args` without a leading execution-policy argument (std::execution::... or execution::...)."""
        if not args:
            return args
        start, end = args[0]
        if self.code.text(start) == "::":
            start += 1
        words = tuple(self.code.text(i) for i in range(start, min(start + 4, end)))
        if words[:2] == ("execution", "::") or words == ("std", "::", "execution", "::"):
            return args[1:]
        return args

    def _standard(self, name: str, args: list[tuple[int, int]], at: int) -> None:
        """Apply the standard-call table of the module docstring (std-qualified, `::`-qualified, or unqualified)."""
        count = len(args)
        if name in ("memcpy", "memmove", "memset") and count == 3:
            self._define(args[0], [args[1]], at)
        elif name in COPIES and count == 3 or name in COPY_FAMILY and count >= 3 or name in SCANS and count >= 3:
            self._define(args[2], args[:2] + args[3:], at)
        elif name == "copy_n" and count == 3:
            self._define(args[2], args[:2], at)
        elif name == "transform" and count in (4, 5):
            target = count - 2
            self._define(args[target], args[:target] + args[target + 1:], at)
        elif name in FILLS and count >= 2:
            self._define(args[0], args[-1:], at)
        elif name == "iota" and count == 3:
            self._define(args[0], args[2:], at)
        elif name == "swap" and count == 2:
            self._define(args[0], args[1:], at)
            self._define(args[1], args[:1], at)
        elif name in ("for_each", "for_each_n") and count >= 2:
            self._for_each(args[0], args[-1], at)

    def _ranges(self, name: str, args: list[tuple[int, int]], at: int) -> None:
        """Apply the std::ranges rows of the module docstring: copy (2 arguments) and transform (3 or 4)."""
        count = len(args)
        if name == "copy" and count == 2:
            self._define(args[1], args[:1], at)
        elif name == "transform" and count == 3:
            self._define(args[1], [args[0], args[2]], at)
        elif name == "transform" and count == 4:
            self._define(args[2], [args[0], args[1], args[3]], at)

    def _member_call(self, callee: Callee, name: str, args: list[tuple[int, int]], at: int) -> None:
        """Read a member call on X that is not an own method: a stream source or sink, or the member table."""
        target = None if callee.receiver is None else self.base_symbol(*callee.receiver)
        if target is None:
            return
        count = len(args)
        types = self.program.symbols[target].types
        if name == "read" and count == 2 and types & IN_STREAMS:
            self._input(args[0], at)
        elif name == "write" and count == 2 and types & OUT_STREAMS:
            self._sink(args[0], at)
        elif name == "assign":
            self._define_symbol(target, self._container_values(args), at)
        elif name in APPENDS or name == "fill" and count == 1:
            self._define_symbol(target, args, at)
        elif name == "insert" and count == 3:
            self._define_symbol(target, self._container_values(args[1:]), at)
        elif name in ("insert", "emplace") or name == "resize" and count == 2:
            self._define_symbol(target, args[1:], at)
        elif name == "swap" and count == 1:
            self._define_symbol(target, args, at)
            other = self.base_symbol(*args[0])
            if other is not None:
                self.raw_defs.append((at, other, Src({target})))

    def _for_each(self, first: tuple[int, int], function: tuple[int, int], at: int) -> None:
        """Link or define the first parameter of the lambda `function` from the range starting at `first`."""
        params = self.body.lambda_params.get(function[0]) if self.code.lambda_at(function[0]) else None
        if not params or params[0] is None:
            return
        param = params[0]
        if self.program.symbols[param].kind == REFERENCE:
            self._link(param, self.base_symbol(*first))
        else:
            self.raw_defs.append((at, param, self.src(*first)))

    def _define(self, target: tuple[int, int], values: Sequence[tuple[int, int]], at: int) -> None:
        """Define base(target) from the mentions of `values` at `at` (nothing when target has no base)."""
        symbol = self.base_symbol(*target)
        if symbol is not None:
            self._define_symbol(symbol, values, at)

    def _define_symbol(self, symbol: int, values: Sequence[tuple[int, int]], at: int) -> None:
        """Define `symbol` from the mentions of `values` at `at`."""
        self.raw_defs.append((at, symbol, Src(parts=[self.src(a, b) for a, b in values])))

    def _input(self, target: tuple[int, int], at: int) -> None:
        """Record an INPUT definition into base(target) at `at`."""
        symbol = self.base_symbol(*target)
        if symbol is not None:
            self.raw_inputs.append((at, symbol))

    def _sink(self, value: tuple[int, int], at: int) -> None:
        """Record an output write at `at` that reads the mentions of `value`."""
        self.raw_sinks.append((at, self.src(*value)))


class _Placer:
    """Places an extractor's events at base positions: token x width, or a named lambda's call (Positions)."""

    def __init__(self, extractor: _Extractor) -> None:
        """Order the named lambdas that run at their calls (_Extractor.placed) and keep their bodies in order."""
        self.extractor = extractor
        body = extractor.body.function.body
        self.width = max(2, body[1] - body[0] + 2)
        self.remaps = sorted(extractor.placed.values())
        self.starts = [start for start, _, _ in self.remaps]

    def bases(self, token: int) -> list[int]:
        """Return the base positions of an event at `token`: one per call of the lambda holding it, else one."""
        k = bisect_right(self.starts, token) - 1
        if k >= 0 and token <= self.remaps[k][1]:
            start, _, sites = self.remaps[k]
            return [site * self.width + (token - start + 1) for site in sites]
        return [token * self.width]

    def place(self) -> None:
        """Fill the extractor's Facts with every event at its base positions, loops included."""
        extractor, facts = self.extractor, self.extractor.facts
        facts.defs = [(p, t, s, src) for t, s, src in extractor.raw_defs for p in self.bases(t)]
        facts.kills = [(p, s) for t, s in extractor.raw_kills for p in self.bases(t)]
        facts.strong = [(p, s, src) for t, s, src in extractor.raw_strong for p in self.bases(t)]
        facts.inputs = [(p, t, s) for t, s in extractor.raw_inputs for p in self.bases(t)]
        facts.sinks = [(p, t, src) for t, src in extractor.raw_sinks for p in self.bases(t)]
        facts.calls = [replace(call, pos=p) for call in extractor.raw_calls for p in self.bases(call.at)]
        facts.loops = [loop for head, end in extractor.body.loops for loop in self._loop(head, end)]

    def _loop(self, head: int, end: int) -> list[tuple[int, int]]:
        """Return the base-position ranges of the loop statement over tokens [head, end]."""
        starts = self.bases(head)
        if starts == [head * self.width]:
            return [(head * self.width, end * self.width + self.width - 1)]
        return [(start, start + end - head) for start in starts]


def _std_qualified(qualifiers: tuple[str, ...]) -> bool:
    """Return True for a std-qualified callee (std:: or ::std::), which never matches an own definition."""
    return qualifiers[:1] == ("std",) or qualifiers[:2] == ("", "std")


def _actual(call: Call, port: Port) -> int | None:
    """Return the caller's symbol a callee port binds to at `call`, or None when it is unbound."""
    tag, number = port
    if tag == "param":
        return call.bases[number] if number < len(call.bases) else None
    if tag == "this":
        return call.this_base
    return number if tag == "global" else None


def _sources(call: Call, port: Port) -> Src | None:
    """Return what a callee port reads at `call` (sources(X)), or None when it is unbound."""
    tag, number = port
    if tag == "param":
        return call.args[number] if number < len(call.args) else None
    if tag == "this":
        return call.this_src
    return Src({number}) if tag == "global" else None


def _has_calls(source: Src) -> bool:
    """Return True when `source` or a part of it holds an own call."""
    seen: set[Src] = set()
    stack = [source]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        if node.calls:
            return True
        stack.extend(node.parts)
    return False


class _Loops:
    """The loop statements of one graph as nested position ranges, with each one's enclosing loop."""

    def __init__(self, ranges: Sequence[tuple[int, int]]) -> None:
        """Order the ranges by head (outer first on a tie) and find each one's parent."""
        items = sorted(set(ranges), key=lambda item: (item[0], -item[1]))
        self.heads = [head for head, _ in items]
        self.ends = [end for _, end in items]
        self.parent: list[int] = []
        stack: list[int] = []
        for index, (head, _) in enumerate(items):
            while stack and self.ends[stack[-1]] < head:
                stack.pop()
            self.parent.append(stack[-1] if stack else -1)
            stack.append(index)

    def chain(self, position: int) -> Iterator[tuple[int, int]]:
        """Yield (head, end) of every loop holding `position`, innermost first."""
        index = bisect_right(self.heads, position) - 1
        while index >= 0 and self.ends[index] < position:
            index = self.parent[index]
        while index >= 0:
            yield self.heads[index], self.ends[index]
            index = self.parent[index]


@dataclass
class _Graph:
    """One function's events with the current summaries applied, as positions (base position x 3 + sub).

    `defs[c]` lists (read, target class, position) of each definition that
    reads class c, sorted by read once the graph is settled (`reads[c]`
    holds the read positions alone, for bisection); `sinks` holds (read,
    classes, token) and `sink_reads[c]` the sorted (read, token) of the
    sinks that read class c; `direct` holds the violations found at a
    call or a sink itself as (read, label, token).
    """

    kills: dict[int, list[int]] = field(default_factory=dict)
    defs: defaultdict[int, list[tuple[int, int, int]]] = field(default_factory=lambda: defaultdict(list))
    reads: dict[int, list[int]] = field(default_factory=dict)
    seeds: list[tuple[int, int, int]] = field(default_factory=list)
    sinks: list[tuple[int, frozenset[int], int]] = field(default_factory=list)
    sink_reads: dict[int, list[tuple[int, int]]] = field(default_factory=dict)
    direct: list[tuple[int, int, int]] = field(default_factory=list)
    touched: set[int] = field(default_factory=set)
    loops: _Loops = field(default_factory=lambda: _Loops(()))

    def settle(self) -> None:
        """Sort each class's definitions and sinks by read position and index the read positions."""
        for cls, items in self.defs.items():
            items.sort()
            self.reads[cls] = [read for read, _, _ in items]
        by_class: defaultdict[int, list[tuple[int, int]]] = defaultdict(list)
        for read, classes, token in self.sinks:
            for cls in classes:
                by_class[cls].append((read, token))
        self.sink_reads = {cls: sorted(items) for cls, items in by_class.items()}

    def visible(self, cls: int, position: int) -> list[tuple[int, int]]:
        """Return the half-open read-position ranges a definition of `cls` at `position` reaches (Visibility)."""
        kills = self.kills.get(cls, ())
        k = bisect_right(kills, position)
        next_kill = kills[k] if k < len(kills) else END
        ranges = [(-END, 0), (position + 1, next_kill)]
        inner = position
        for head, end in self.loops.chain(position):
            if end >= next_kill:
                break
            f = bisect_right(kills, head)
            first_kill = kills[f] if f < len(kills) else END
            top = min(inner, first_kill)
            if head < top:
                ranges.append((head, top))
            inner = head
        return ranges


def _skip_next(fired: dict[int, int], i: int) -> int:
    """Return the first index from `i` that `fired` does not skip (skip pointers with path compression)."""
    root = i
    while root in fired:
        root = fired[root]
    while i in fired and fired[i] != root:
        fired[i], i = root, fired[i]
    return root


def _reach_labels(graph: _Graph, seeds: Sequence[tuple[int, int, int]], budget: Budget | None) -> dict[State, int]:
    """Return the label of each state reached from `seeds` ((class, position, label)), the smallest that reaches it.

    Seeds are visited in label order and a state keeps the first label
    that reaches it; each definition fires once, for the first state that
    sees it, so the reach costs about the number of definitions.
    """
    labels: dict[State, int] = {}
    fired: dict[int, dict[int, int]] = {}
    queue: deque[State] = deque()
    for cls, pos, label in sorted(seeds, key=lambda seed: (seed[2], seed[1], seed[0])):
        if (cls, pos) in labels:
            continue
        labels[(cls, pos)] = label
        queue.append((cls, pos))
        while queue:
            source, position = queue.popleft()
            reads = graph.reads.get(source)
            if not reads:
                continue
            skip = fired.setdefault(source, {})
            for lo, hi in graph.visible(source, position):
                i, stop = _skip_next(skip, bisect_left(reads, lo)), bisect_left(reads, hi)
                _spend(budget, 1)
                while i < stop:
                    skip[i] = i + 1
                    _, target, defined = graph.defs[source][i]
                    if (target, defined) not in labels:
                        labels[(target, defined)] = label
                        queue.append((target, defined))
                    i = _skip_next(skip, i + 1)
                    _spend(budget, 1)
    return labels


def _reach_masks(graph: _Graph, seeds: Sequence[tuple[int, int, int]], budget: Budget | None) -> dict[State, int]:
    """Return the set of seed bits (a mask) of each state reached from `seeds` ((class, position, bit)).

    Each definition fires once per bit that reaches a state seeing it, so
    one pass reaches every port of a summary at once.
    """
    masks: dict[State, int] = {}
    pending: dict[State, int] = {}
    queue: deque[State] = deque()
    for cls, pos, bit in seeds:
        state = (cls, pos)
        new = bit & ~masks.get(state, 0)
        if new:
            masks[state] = masks.get(state, 0) | new
            if state not in pending:
                queue.append(state)
            pending[state] = pending.get(state, 0) | new
    fired: dict[tuple[int, int], dict[int, int]] = {}
    while queue:
        state = queue.popleft()
        delta = pending.pop(state)
        source, position = state
        reads = graph.reads.get(source)
        if not reads:
            continue
        ranges = [(bisect_left(reads, lo), bisect_left(reads, hi)) for lo, hi in graph.visible(source, position)]
        for bit in _bits(delta):
            skip = fired.setdefault((source, bit), {})
            for start, stop in ranges:
                i = _skip_next(skip, start)
                _spend(budget, 1)
                while i < stop:
                    skip[i] = i + 1
                    _, target, defined = graph.defs[source][i]
                    reached = (target, defined)
                    if not masks.get(reached, 0) & bit:
                        masks[reached] = masks.get(reached, 0) | bit
                        if reached not in pending:
                            queue.append(reached)
                        pending[reached] = pending.get(reached, 0) | bit
                    i = _skip_next(skip, i + 1)
                    _spend(budget, 1)
    return masks


def _bits(mask: int) -> Iterator[int]:
    """Yield each set bit of `mask` as a power of two, lowest first."""
    while mask:
        low = mask & -mask
        yield low
        mask ^= low


def _sees_sink(graph: _Graph, cls: int, position: int) -> tuple[int, int] | None:
    """Return the first (read, token) of a sink that reads `cls` visibly from a definition at `position`, or None."""
    sinks = graph.sink_reads.get(cls)
    if not sinks:
        return None
    best: tuple[int, int] | None = None
    for lo, hi in graph.visible(cls, position):
        i = bisect_left(sinks, (lo, -END))
        if i < len(sinks) and sinks[i][0] < hi and (best is None or sinks[i] < best):
            best = sinks[i]
    return best


def _first_violation(graph: _Graph, labels: Mapping[State, int], budget: Budget | None) -> tuple[int, int, int] | None:
    """Return (sink read, source label, sink token) of the violation with the smallest sink, then source, or None.

    For each reached state the first sink in each of its visible ranges is
    a candidate; the least candidate is the least visible pair.
    """
    best: tuple[int, int, int] | None = None
    for (cls, position), label in labels.items():
        _spend(budget, 1)
        found = _sees_sink(graph, cls, position)
        if found is not None and (best is None or (found[0], label) < best[:2]):
            best = (found[0], label, found[1])
    return best


class _Analyzer:
    """The state of one analyze() call: every function's facts, kill sets, and flow summaries."""

    def __init__(self, program: Program, device_reads: frozenset[str]) -> None:
        """Extract the facts of every function that is not a `lassi_io_` definition."""
        self.program = program
        self.budget = program.budget
        analyzed = [i for i, item in enumerate(program.functions) if not item.name.startswith(HARNESS_PREFIX)]
        own: dict[str, list[int]] = defaultdict(list)
        for index in analyzed:
            own[program.functions[index].name].append(index)
        self.facts = {index: _Extractor(program, index, own, device_reads).run() for index in analyzed}
        self.callers: defaultdict[int, set[int]] = defaultdict(set)
        for index, facts in self.facts.items():
            for call in facts.calls:
                for callee in call.callees:
                    self.callers[callee].add(index)
        self.kills_of: dict[int, frozenset[Port]] = {index: frozenset() for index in self.facts}
        self.flows_of: dict[int, frozenset[tuple[Port, Port]]] = {index: frozenset() for index in self.facts}
        self.strong_kills: dict[int, list[tuple[int, int]]] = {index: [] for index in self.facts}
        self.dependent = any(_has_calls(right) for facts in self.facts.values() for _, _, right in facts.strong)

    def run(self) -> Analysis:
        """Return the Analysis: the sinks, and the witness once the summaries have settled."""
        found = {
            (self.program.functions[index].rank, token): self._site(index, token)
            for index, facts in self.facts.items() for _, token, _ in facts.sinks
        }
        sinks = tuple(found[key] for key in sorted(found))
        if not sinks:
            return Analysis((), None)
        self._settle()
        return Analysis(sinks, self._violation())

    def _site(self, index: int, at: int) -> Site:
        """Return the Site of token `at` in function `index`'s file."""
        rank = self.program.functions[index].rank
        return Site(rank, at, self.program.paths[rank], self.program.codes[rank].tokens[at].line)

    def _settle(self) -> None:
        """Run both fixpoints, then retake the strong-update decisions from the summaries until they hold.

        Each retake keeps only kills that were decided before, so the
        decisions only shrink and the loop ends.
        """
        decided = self._decide(None)
        while True:
            self.strong_kills = decided
            self.kills_of = {index: frozenset() for index in self.facts}
            self.flows_of = {index: frozenset() for index in self.facts}
            self._fixpoint(self._kill_step)
            self._fixpoint(self._flow_step)
            if not self.dependent:
                return
            again = self._decide(decided)
            if again == decided:
                return
            decided = again

    def _decide(self, previous: Mapping[int, list[tuple[int, int]]] | None) -> dict[int, list[tuple[int, int]]]:
        """Return each function's strong-update kills: those whose right side does not read the target's class."""
        decided: dict[int, list[tuple[int, int]]] = {}
        for index, facts in self.facts.items():
            kept = None if previous is None else set(previous[index])
            decided[index] = [
                (pos, symbol)
                for pos, symbol, right in facts.strong
                if (kept is None or (pos, symbol) in kept)
                and facts.find(symbol) not in self._read_classes(facts, right)
            ]
        return decided

    def _read_classes(self, facts: Facts, source: Src) -> set[int]:
        """Return the classes `source` reads with the current summaries (own calls read their RET sources)."""
        classes: set[int] = set()
        seen: set[Src] = set()
        stack = [source]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            _spend(self.budget, 1)
            classes.update(facts.find(symbol) for symbol in node.symbols)
            stack.extend(node.parts)
            for call in node.calls:
                for callee in call.callees:
                    for port, target in self.flows_of[callee]:
                        if target == RET_PORT and (found := _sources(call, port)) is not None:
                            stack.append(found)
        return classes

    def _fixpoint(self, step: Callable[[int], bool]) -> None:
        """Run `step(index) -> changed` over a worklist of functions until nothing changes; callers rerun.

        Only functions that some function calls are stepped: a summary is
        read only at a call, so main and every uncalled function keep none.
        """
        queue = deque(index for index in sorted(self.facts) if self.callers.get(index))
        queued = set(queue)
        while queue:
            index = queue.popleft()
            queued.discard(index)
            if step(index):
                for caller in sorted(self.callers[index]):
                    if caller not in queued:
                        queue.append(caller)
                        queued.add(caller)

    def _ports(self, index: int, touched: set[int]) -> list[tuple[Port, int, bool]]:
        """Return (port, symbol, out-port) for each parameter, THIS, and touched file-scope variable."""
        body = self.facts[index].body
        symbols = self.program.symbols
        ports = [
            (("param", i), symbol, symbols[symbol].out) for i, symbol in enumerate(body.params) if symbol is not None
        ]
        if body.this is not None:
            ports.append((THIS_PORT, body.this, True))
        ports.extend((("global", symbol), symbol, True) for symbol in sorted(touched))
        return ports

    def _own_kills(self, index: int) -> list[tuple[int, int]]:
        """Return function `index`'s own kills (device reads) and its decided strong-update kills."""
        return self.facts[index].kills + self.strong_kills[index]

    def _kill_step(self, index: int) -> bool:
        """Recompute function `index`'s killed ports; return True when they grew."""
        facts = self.facts[index]
        killed = {facts.find(symbol) for _, symbol in self._own_kills(index)}
        touched = set(facts.globals_used)
        for call in facts.calls:
            for callee in call.callees:
                _spend(self.budget, len(self.kills_of[callee]) + 1)
                for port in self.kills_of[callee]:
                    actual = _actual(call, port)
                    if actual is not None:
                        killed.add(facts.find(actual))
                    if port[0] == "global":
                        touched.add(port[1])
        ports = self._ports(index, touched)
        new = self.kills_of[index] | {port for port, symbol, out in ports if out and facts.find(symbol) in killed}
        changed = new != self.kills_of[index]
        self.kills_of[index] = frozenset(new)
        return changed

    def _flow_step(self, index: int) -> bool:
        """Recompute function `index`'s flow summary; return True when it grew."""
        graph = self._graph(index)
        new = self.flows_of[index] | self._summary(index, graph)
        changed = new != self.flows_of[index]
        self.flows_of[index] = new
        return changed

    def _graph(self, index: int, file_scope: bool = False) -> _Graph:
        """Return function `index`'s events with the current summaries applied (see the module docstring).

        With `file_scope`, the graph also holds the file-scope initializers'
        definitions (_file_scope), as each main's graph does.
        """
        facts = self.facts[index]
        graph = _Graph(touched=set(facts.globals_used), loops=_Loops([(h * 3, e * 3 + 2) for h, e in facts.loops]))
        kills: defaultdict[int, list[int]] = defaultdict(list)
        for pos, symbol in self._own_kills(index):
            kills[facts.find(symbol)].append(pos * 3 + 1)
        for call in facts.calls:
            for callee in call.callees:
                for port in self.kills_of[callee]:
                    actual = _actual(call, port)
                    if actual is not None:
                        kills[facts.find(actual)].append(call.pos * 3 + 1)
                    self._touch(graph, port)
        graph.kills = {cls: sorted(positions) for cls, positions in kills.items()}
        _spend(self.budget, len(facts.defs) + len(facts.calls) + len(facts.sinks) + 1)
        for pos, _, target, source in facts.defs:
            self._add_def(graph, facts, facts.find(target), pos, source)
        for pos, token, symbol in facts.inputs:
            graph.seeds.append((facts.find(symbol), pos * 3 + 2, token))
        for pos, token, source in facts.sinks:
            self._add_sink(graph, facts, pos, token, source)
        for call in facts.calls:
            self._apply(graph, facts, call)
        if file_scope:
            self._file_scope(graph, facts)
        graph.settle()
        return graph

    @staticmethod
    def _touch(graph: _Graph, port: Port) -> None:
        """Note a file-scope variable port as touched by the function."""
        if port[0] == "global":
            graph.touched.add(port[1])

    def _apply(self, graph: _Graph, facts: Facts, call: Call) -> None:
        """Add the events of each matched callee's flow summary at `call` (see the module docstring)."""
        for callee in call.callees:
            _spend(self.budget, len(self.flows_of[callee]) + 1)
            for source, target in self.flows_of[callee]:
                if target == RET_PORT:
                    continue
                self._touch(graph, source)
                self._touch(graph, target)
                if target == SINK_PORT:
                    if source == INPUT_PORT:
                        graph.direct.append((call.pos * 3, call.at, call.at))
                    elif (found := _sources(call, source)) is not None:
                        self._add_sink(graph, facts, call.pos, call.at, found)
                    continue
                actual = _actual(call, target)
                if actual is None:
                    continue
                if source == INPUT_PORT:
                    graph.seeds.append((facts.find(actual), call.pos * 3 + 2, call.at))
                elif (found := _sources(call, source)) is not None:
                    self._add_def(graph, facts, facts.find(actual), call.pos, found)

    def _add_def(self, graph: _Graph, facts: Facts, target: int, pos: int, source: Src) -> None:
        """Add a definition of class `target` at base position `pos` from `source`, and a seed when it carries input."""
        classes, origin = self._evaluate(graph, facts, source)
        for cls in classes:
            graph.defs[cls].append((pos * 3, target, pos * 3 + 2))
        if origin is not None:
            graph.seeds.append((target, pos * 3 + 2, origin))

    def _add_sink(self, graph: _Graph, facts: Facts, pos: int, token: int, source: Src) -> None:
        """Add an output write at base position `pos` (token `token`) that reads `source`; its input is a violation."""
        classes, origin = self._evaluate(graph, facts, source)
        graph.sinks.append((pos * 3, frozenset(classes), token))
        if origin is not None:
            graph.direct.append((pos * 3, origin, token))

    def _evaluate(self, graph: _Graph, facts: Facts, source: Src) -> tuple[set[int], int | None]:
        """Return the classes `source` reads with the current summaries, and the first call that gives it input."""
        classes: set[int] = set()
        origin: int | None = None
        seen: set[Src] = set()
        stack = [source]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            _spend(self.budget, 1)
            classes.update(facts.find(symbol) for symbol in node.symbols)
            stack.extend(node.parts)
            for call in node.calls:
                for callee in call.callees:
                    for port, target in self.flows_of[callee]:
                        if target != RET_PORT:
                            continue
                        self._touch(graph, port)
                        if port == INPUT_PORT:
                            origin = call.at if origin is None else min(origin, call.at)
                        elif (found := _sources(call, port)) is not None:
                            stack.append(found)
        return classes, origin

    def _summary(self, index: int, graph: _Graph) -> frozenset[tuple[Port, Port]]:
        """Return the flows of function `index` with its graph, every port reached in one pass (Own functions)."""
        facts = self.facts[index]
        ports = self._ports(index, graph.touched)
        named: list[Port] = [port for port, _, _ in ports] + [INPUT_PORT]
        input_bit = 1 << len(ports)
        seeds = [(facts.find(symbol), ENTRY, 1 << n) for n, (_, symbol, _) in enumerate(ports)]
        seeds.extend((cls, pos, input_bit) for cls, pos, _ in graph.seeds)
        masks = _reach_masks(graph, seeds, self.budget)
        outs: defaultdict[int, list[Port]] = defaultdict(list)
        for port, symbol, out in ports:
            if out:
                outs[facts.find(symbol)].append(port)
        ret = facts.find(facts.body.ret)
        flows: set[tuple[Port, Port]] = set()
        for (cls, pos), mask in masks.items():
            _spend(self.budget, 1)
            sources = [named[bit.bit_length() - 1] for bit in _bits(mask)]
            if _sees_sink(graph, cls, pos) is not None:
                flows.update((port, SINK_PORT) for port in sources)
            if pos == ENTRY:
                continue
            if cls == ret:
                flows.update((port, RET_PORT) for port in sources)
            kills = graph.kills.get(cls, ())
            if cls in outs and (not kills or kills[-1] < pos):
                flows.update((port, out) for port in sources for out in outs[cls] if out != port)
        if graph.direct:
            flows.add((INPUT_PORT, SINK_PORT))
        if self._constructor(index):
            for i, symbol in enumerate(facts.body.params):
                if symbol is not None:
                    flows.update({(("param", i), RET_PORT), (("param", i), THIS_PORT)})
        return frozenset(flows)

    def _constructor(self, index: int) -> bool:
        """Return True for a constructor: a member function named as its class."""
        function: Function = self.program.functions[index]
        return function.member and function.name == function.class_name

    def _is_main(self, index: int) -> bool:
        """Return True for the program's main: a free function named main at file scope."""
        function: Function = self.program.functions[index]
        return function.name == "main" and not function.member and not function.qualifiers and not function.namespaced

    def _violation(self) -> tuple[Site, Site] | None:
        """Return the witness (source, sink) of every main's graph, or of every function's when there is no main."""
        mains = [index for index in sorted(self.facts) if self._is_main(index)]
        best: tuple[tuple[int, int, int], int, int, int] | None = None
        for index in mains or sorted(self.facts):
            graph = self._graph(index, file_scope=bool(mains))
            labels = _reach_labels(graph, graph.seeds, self.budget)
            rank = self.program.functions[index].rank
            for found in [_first_violation(graph, labels, self.budget), *graph.direct]:
                if found is None:
                    continue
                read, label, token = found
                if best is None or (rank, read, label) < best[0]:
                    best = ((rank, read, label), index, label, token)
        if best is None:
            return None
        _, index, label, token = best
        return self._site(index, label), self._site(index, token)

    def _file_scope(self, graph: _Graph, facts: Facts) -> None:
        """Add the definitions of value- or array-declared file-scope initializers, visible everywhere."""
        program = self.program
        for decl in program.global_decls:
            found = decl.declarator
            code = program.codes[decl.rank]
            if found.init is None or found.kind not in (VALUE, ARRAY):
                continue
            target = facts.find(program.globals[code.text(found.name)])
            for i in code.iter_mentions(*found.init, qualified=program.globals):
                symbol = program.globals.get(code.text(i))
                if symbol is not None:
                    graph.defs[facts.find(symbol)].append((ENTRY - 1, target, ENTRY))
