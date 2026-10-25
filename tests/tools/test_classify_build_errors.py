"""Tests for task P17.24: tools/classify_build_errors.py, the cpp -> tt build errors of run records in buckets.

Bible: Result Record (Trial, Attempt, Diagnostic, Request; Storage), Agent
Rules 1 (every count is copied from a record, never a measurement) and 5 (no
prompt, reply, or reference text leaves the records). plans/p17-portable.md,
task P17.24, added on 2026-10-25 from the passes of 2026-10-23 and 2026-10-24
(results/rngd-sweep5-20261024, results/rngd-matrix-short-20261024).

The interface these tests fix:

- abstract_message(message) -> str returns a message's bucket, by this fixed
  rule, applied in this order:
  1. Symbols. The rest of the message after the first "undefined symbol: "
     or "duplicate symbol: " (ld.lld) or "undefined reference to " (GNU
     ld), each space there standing for any run of whitespace, becomes <S>
     when it holds a character other than whitespace (a parsed diagnostic
     is one line, so that is the rest of the line). Then each remaining C++
     qualified name, a maximal run of ASCII letters, digits, _, and :: that
     holds ::, and each identifier (an ASCII letter or _, then ASCII
     letters, digits, and _) that no letter, digit, or _ precedes and that
     ( directly follows, becomes <S>.
  2. Quoted tokens. Each span that opens with ' and closes at the next ',
     opens with " and closes at the next ", opens with ` and closes at the
     next ' (the linker's quoting), or opens with U+2018 and closes at the
     next U+2019 (U+201C and U+201D likewise), the quotes included, becomes
     <Q>. A quote that never closes stays as it is.
  3. Paths. Each maximal run of characters other than whitespace, the quote
     characters of step 2, and the characters : , ; ( ) that holds at least
     one /, or that ends in .cpp, .cc, .c, .hpp, .h, or .hh after at least
     one other character (a bare source file name), becomes <P>.
  4. Numbers. Each maximal run of decimal digits, with an optional . and
     more digits, or 0x or 0X and hex digits, that no identifier character
     (an ASCII letter or digit, _, $, or any character outside ASCII)
     touches on either side, becomes <N>. So the digits of an identifier such
     as uint32_t or c_16 stay here, and a sign before a number stays.
  5. Other tokens. Each whitespace-separated token, its leading and trailing
     punctuation (the characters , . ; : ( ) [ ] { } ' " ! ? and `) set
     aside, becomes <X>, its punctuation kept around it, unless what is left
     is a plain lowercase ASCII word ([a-z]+, or such words joined by -, as
     in out-of-line) or, the placeholders <Q>, <P>, <N>, <S>, and <X> aside,
     holds only ASCII punctuation other than _ and $ (as in <P>:<N>, -<N>,
     or >>>). So any token holding a letter, digit, _, $, or non-ASCII
     character outside the placeholders that is not a plain lowercase ASCII
     word is masked, whatever the message's wording.
  6. Each run of whitespace becomes one space, and both ends are stripped.
  ABSTRACTIONS below pins the rule with examples; it is idempotent. A bucket
  is plain ASCII, and what can survive in it is the placeholders, plain
  lowercase ASCII words, and tokens of ASCII punctuation alone (for example
  >>> or <N>.<N>). The one known limit: a plain lowercase ASCII word the
  model chose (an identifier, or a word of #error text) stays.
- classify(runs_root) returns an object whose mappings are each read here
  with dict(), an inner mapping that is empty counting as absent:
  overall (bucket -> count), per_run (run id -> bucket -> count),
  per_project and per_arm (the trial_id's first and second segment ->
  bucket -> count), per_item (item -> bucket -> count), per_attempt
  (attempt index, an int -> bucket -> count), and recurs ((run id,
  trial_id) -> bool); and provenance (run id -> the distinct provenance
  of its records, each with commit, dirty, device, sdk, and date from the
  record's provenance copy and pins, the (name, pin) of each toolchain pin
  the record holds, in the order first read). The records it reads are
  the trial.json files at a trial's depth under <runs root>/runs/<run id>/,
  one directory level per trial_id segment, as the runner writes them, so no
  file in a build directory is read; the run id is the directory under
  runs/. It keeps the trials whose bench_item.direction is cpp-tt, and in
  every attempt of each it buckets every Diagnostic whose stage is compile
  and whose severity is error (a warning, a note, and a parse-, run-, or
  jit-stage diagnostic are not bucketed). A count is a number of such
  diagnostics. recurs holds every kept trial and no other: True when, for
  some k, a bucket of attempt k is also a bucket of attempt k + 1. A record
  that is not JSON, or that the strict record loader refuses, raises
  ValueError (or a subclass) naming its path; nothing is skipped silently.
- main(argv) -> int runs `classify_build_errors.py <runs root>`: it prints
  the report on stdout and returns 0, each overall count on a line with its
  bucket, every kept trial's id, every run id, project, and arm, and each
  run's commit, date, and toolchain pins named; a malformed record gives 2,
  a message on stderr naming the record's path, and no report.
- The report holds counts, bucket patterns, run ids, projects, arms, items,
  attempt indexes, trial ids, and the runs' provenance only: never a prompt,
  a reply, a file the model wrote, a reference's text, the trial's context,
  a diagnostic's raw message, or its file. The tool writes nothing and
  starts no process.

The run trees are built in a temporary directory with the runner's own
writer, lassi.core.store.write_trial. Every trial, message, prompt, reply,
commit, and date here is SYNTHETIC or a PLACEHOLDER fixture value, and every
expected count is the count of the SYNTHETIC diagnostics written here, never
a measurement.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from lassi.core.interfaces import Sampling
from lassi.core.record import (
    Attempt,
    BenchItem,
    Context,
    Diagnostic,
    Final,
    ModelInfo,
    Provenance,
    Request,
    RequestMessage,
    ToolchainPins,
    Trial,
    make_trial_id,
)
from lassi.core.store import TRIAL_JSON, TextStore, trial_dir, write_trial

REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "tools" / "classify_build_errors.py"

# ---------------------------------------------------------------------------
# The rule (abstract_message)

# The typographic quotes a compiler prints in a UTF-8 locale (U+2018, U+2019, U+201C, U+201D), built from
# their code points so this file stays plain ASCII.
LSQ, RSQ, LDQ, RDQ = chr(0x2018), chr(0x2019), chr(0x201C), chr(0x201D)
# U+2118 and U+203F, which a C++ identifier may hold and which are neither letters nor digits to str.isalnum.
SCRIPT_P, UNDERTIE = chr(0x2118), chr(0x203F)
# Each example: a message as a compiler or this project writes it, and its bucket.
ABSTRACTIONS = [
    ("use of undeclared identifier 'CreateCircularBuffer'", "use of undeclared identifier <Q>"),
    ("no member named 'EnqueueWriteMeshBuffer' in namespace 'tt::tt_metal'", "no member named <Q> in namespace <Q>"),
    (
        "'cb_wait_front' was not declared in this scope; did you mean 'cb_wait_front_tiles'?",
        "<Q> was not declared in this scope; did you mean <Q>?",
    ),
    ("tt_metal/host_api.hpp: No such file or directory", "<P>: <X> such file or directory"),
    ("In file included from /usr/include/c++/11/vector:64,", "<X> file included from <P>:<N>,"),
    ("scale_ref.h: No such file or directory", "<P>: <X> such file or directory"),
    (
        "the expected file 'scale.cpp' is missing: "
        "return it in a fenced block whose first line is `// FILE: scale.cpp`",
        "the expected file <Q> is missing: return it in a fenced block whose first line is `<P> <X>: <P>`",
    ),
    # Paths that hold both quote kinds: an unclosed quote ends a path run, and what is left is masked (step 5).
    ('/tmp/a"b/c\'d.h: No such file or directory', '<P>"<P>\'<P>: <X> such file or directory'),
    ('/tmp/it\'s "x"/scale.cpp: No such file or directory', "<X> <P>: <X> such file or directory"),
    # The linkers' symbols: GNU ld's quoted form, and ld.lld's unquoted forms, as the driver pattern reads them
    # (lassi.toolchains.ttmetal_build) and as printed.
    (
        "/scratch/runs/x/attempt03/build/scale.cpp:21: undefined reference to `main'",
        "<P>:<N>: undefined reference to <S>",
    ),
    ("ld returned 1 exit status", "ld returned <N> exit status"),
    ("undefined reference to `helper(float*, int)'", "undefined reference to <S>"),
    ("undefined symbol: helper(int)", "undefined symbol: <S>"),
    ("ld.lld: error: undefined symbol: helper(int)", "<X>: error: undefined symbol: <S>"),
    ("duplicate symbol: tt::tt_metal::detail::Finish(tt::tt_metal::CommandQueue&)", "duplicate symbol: <S>"),
    (
        "undefined symbol: std::vector<unsigned int, std::allocator<unsigned int>> "
        "tt::tt_metal::CreateBuffer<float>(tt::tt_metal::IDevice*, unsigned long)",
        "undefined symbol: <S>",
    ),
    # Symbols after a phrase step 1 does not know, masked by step 5 whatever the wording.
    ("undefined hidden symbol: g_secretTable", "undefined hidden symbol: <X>"),
    ("undefined hidden symbol: secret_helper<float>(float)", "undefined hidden symbol: <X>)"),
    (
        "relocation refers to a symbol in a discarded section: secretInline",
        "relocation refers to a symbol in a discarded section: <X>",
    ),
    ("kernel.cu: No such file or directory", "<X>: <X> such file or directory"),
    # $ and every character outside ASCII are identifier characters, so tokens holding them are masked.
    ("error: $foo $$", "error: <X> <X>"),
    (f"undefined hidden symbol: {SCRIPT_P}{UNDERTIE}{SCRIPT_P}", "undefined hidden symbol: <X>"),
    # Qualified names and call shapes outside a symbol phrase.
    ("no matching call to tt::tt_metal::CreateKernel(program, 0)", "no matching call to <X>, <N>)"),
    # The apostrophe of free text pairs with the next quote; what is left is masked, so x does not stay.
    ("can't find 'x'", "<X>'"),
    # The known limit: the plain lowercase words of a text the model wrote, such as a #error line, stay.
    ("#error unsupported arch 5", "<X> unsupported arch <N>"),
    ('cannot open source file "kernels/scale.h"', "cannot open source file <Q>"),
    (
        "too many arguments to function 'void f(int, int)'; expected 2, have 3",
        "too many arguments to function <Q>; expected <N>, have <N>",
    ),
    ("array subscript 16 is above array bounds of 'uint32_t [16]'", "array subscript <N> is above array bounds of <Q>"),
    ("conversion from uint64_t to uint32_t may change value", "conversion from <X> to <X> may change value"),
    ("c_16 needs 2 tiles", "<X> needs <N> tiles"),
    ("CBIndex::c_16 needs 2 tiles", "<S> needs <N> tiles"),
    ("requested alignment 0x20 is not a power of 2", "requested alignment <N> is not a power of <N>"),
    ("static assertion failed: tile size is 1.5 times 32", "static assertion failed: tile size is <N> times <N>"),
    ("value -1 is out of range", "value -<N> is out of range"),
    (f"{LSQ}CreateKernel{RSQ} was not declared in this scope", "<Q> was not declared in this scope"),
    (f"{LDQ}kernels/reader.cpp{RDQ} not found", "<Q> not found"),
    ("expected   primary-expression\tbefore ')' token ", "expected primary-expression before <Q> token"),
    ("missing terminating ' character", "missing terminating ' character"),
    ("", ""),
]

# Pairs of messages that differ only in quoted tokens, paths, numbers, and masked tokens, so they share a bucket.
SAME_BUCKET = [
    ("use of undeclared identifier 'CreateKernel'", "use of undeclared identifier 'cb_reserve_back'"),
    ("tt_metal/a.hpp: No such file or directory", "/opt/x/b/c.h: No such file or directory"),
    ("expected 2 arguments, have 7", "expected 10 arguments, have 0x3"),
    ("undefined hidden symbol: g_secretTable", "undefined hidden symbol: OtherTable"),
]
# Pairs that differ in a word, so they do not.
OTHER_BUCKET = [
    ("use of undeclared identifier 'x'", "use of undeclared type 'x'"),
    ("expected 2 arguments", "expected 2 operands"),
]


@lru_cache(maxsize=1)
def tool() -> ModuleType:
    """Load tools/classify_build_errors.py as a module, failing with the task's name while it does not exist."""
    assert TOOL.is_file(), f"{TOOL.relative_to(REPO).as_posix()} does not exist yet (task P17.24)"
    spec = importlib.util.spec_from_file_location("classify_build_errors_under_test", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # registered before it runs, as an import would be
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(("message", "bucket"), ABSTRACTIONS, ids=[f"ex{i:02d}" for i in range(len(ABSTRACTIONS))])
def test_the_rule_abstracts_quoted_tokens_paths_and_numbers(message: str, bucket: str) -> None:
    abstract = tool().abstract_message
    assert abstract(message) == bucket
    assert abstract(bucket) == bucket, "the rule is idempotent: a bucket is its own bucket"


@pytest.mark.parametrize(("first", "second"), SAME_BUCKET)
def test_messages_that_differ_only_in_what_the_rule_replaces_share_a_bucket(first: str, second: str) -> None:
    abstract = tool().abstract_message
    assert abstract(first) == abstract(second)


@pytest.mark.parametrize(("first", "second"), OTHER_BUCKET)
def test_messages_that_differ_in_a_word_do_not_share_a_bucket(first: str, second: str) -> None:
    abstract = tool().abstract_message
    assert abstract(first) != abstract(second)


# ---------------------------------------------------------------------------
# The SYNTHETIC run trees

PROJECT, ARM, SUITE = "syn-sweep", "syn-model", "syn-pairs"
# Run B's arm: a run binds one arm, so the per-arm counts differ from the overall ones.
ARM_B = "syn-model-b"
CPP_TT, TT_CPP = "cpp-tt", "tt-cpp"
RUN_A, RUN_B = "syn-run-a", "syn-run-b"
ITEM_A, ITEM_B = "item_a", "item_b"
# A SYNTHETIC commit id and a PLACEHOLDER date; neither is a commit or a date of any run.
SYNTHETIC_COMMIT = "0123456789abcdef0123456789abcdef01234567"
PLACEHOLDER_DATE = "2026-10-25T00:00:00+00:00"
SYNTHETIC_RECIPE_HASH = hashlib.sha256(b"SYNTHETIC recipe\n").hexdigest()
# PLACEHOLDER toolchain pins, in ToolchainPins field order; no pin of any toolchain.
PLACEHOLDER_PINS = (("tt_metal", "PLACEHOLDER-tt-metal-pin"), ("gcc", "PLACEHOLDER-gcc-pin"))

# Texts that sit in the records and must never reach the report: each prompt (which carries the source reference,
# as a generate prompt does), each reply, each file the model wrote, the trial's context, and each diagnostic's
# file. IDENT_MARKER is an identifier of the replies that a compile error quotes.
PROMPT_MARKER = "PromptMarkerQv"
REFERENCE_MARKER = "ReferenceMarkerQv"
REPLY_MARKER = "ReplyMarkerQv"
FILES_MARKER = "FilesMarkerQv"
CONTEXT_MARKER = "ContextMarkerQv"
IDENT_MARKER = "IdentMarkerQv"
DIAGNOSTIC_FILE = "filemarkerqv/scale.cpp"
MARKERS = (PROMPT_MARKER, REFERENCE_MARKER, REPLY_MARKER, FILES_MARKER, CONTEXT_MARKER, IDENT_MARKER, DIAGNOSTIC_FILE)

# The buckets the kept trials' errors fall in, and the one only a tt -> cpp trial holds.
UNDECLARED = "use of undeclared identifier <Q>"
NO_MEMBER = "no member named <Q> in namespace <Q>"
NO_FILE = "<P>: <X> such file or directory"
TOO_MANY = "too many arguments to function <Q>; expected <N>, have <N>"
LINKER = "ld returned <N> exit status"


def compile_error(message: str) -> Diagnostic:
    """Return a compile-stage error at a SYNTHETIC place in the model's file."""
    return Diagnostic(stage="compile", severity="error", file=DIAGNOSTIC_FILE, line=12, column=5, message=message)


# Diagnostics no bucket takes: a compile warning and note, and errors and warnings of other stages.
IGNORED = (
    Diagnostic(stage="compile", severity="warning", message="unused variable 'cb'"),
    Diagnostic(stage="compile", severity="note", message="declared here 'cb'"),
    Diagnostic(stage="run", severity="error", code="run-error", message="the program exited with status 3"),
    Diagnostic(stage="jit", severity="error", message="kernel 'reader' failed to compile at line 7"),
    Diagnostic(stage="parse", severity="warning", code="unfenced-file", message="SYNTHETIC unfenced line 4"),
)
IGNORED_BUCKETS = (
    "unused variable <Q>",
    "declared here <Q>",
    "the program exited with status <N>",
    "kernel <Q> failed to compile at line <N>",
    "<X> unfenced line <N>",
)


@dataclass(frozen=True)
class Plan:
    """One SYNTHETIC trial: its run directory, direction, item, run number, each attempt's diagnostics, and arm."""

    run_id: str
    direction: str
    item: str
    run: int
    attempts: tuple[tuple[Diagnostic, ...], ...]
    arm: str = ARM

    @property
    def trial_id(self) -> str:
        """Return the trial's id."""
        return make_trial_id(PROJECT, self.arm, SUITE, self.direction, self.item, self.run)


T1 = Plan(RUN_A, CPP_TT, ITEM_A, 1, (
    (compile_error("use of undeclared identifier 'CreateCircularBuffer'"),
     compile_error("no member named 'EnqueueWriteMeshBuffer' in namespace 'tt::tt_metal'"), *IGNORED),
    (compile_error(f"use of undeclared identifier '{IDENT_MARKER}'"),),
    (),
))
T2 = Plan(RUN_A, CPP_TT, ITEM_B, 1, (
    (compile_error("tt_metal/host_api.hpp: No such file or directory"),),
    (compile_error("too many arguments to function 'void f(int, int)'; expected 2, have 3"),),
    (compile_error("tt_metal/impl/device/device.hpp: No such file or directory"),),
))
# The tt -> cpp trial, whose bucket recurs and is left out with it.
T3 = Plan(RUN_A, TT_CPP, ITEM_A, 1, (
    (compile_error("ld returned 1 exit status"),),
    (compile_error("ld returned 1 exit status"),),
))
T4 = Plan(RUN_B, CPP_TT, ITEM_A, 1, (
    (compile_error("use of undeclared identifier 'cb_wait_front'"),
     compile_error("use of undeclared identifier 'cb_pop_front'")),
    (compile_error("no member named 'CreateBuffer' in namespace 'tt::tt_metal::distributed'"),),
    (compile_error("no member named 'Finish' in namespace 'tt::tt_metal'"), *IGNORED),
), ARM_B)
T5 = Plan(RUN_B, CPP_TT, ITEM_B, 2, ((IGNORED[0],),), ARM_B)
PLANS = (T1, T2, T3, T4, T5)
KEPT = (T1, T2, T4, T5)

# The expected classification of PLANS, counted by hand from the diagnostics above.
OVERALL = {UNDECLARED: 4, NO_MEMBER: 3, NO_FILE: 2, TOO_MANY: 1}
PER_RUN = {
    RUN_A: {UNDECLARED: 2, NO_MEMBER: 1, NO_FILE: 2, TOO_MANY: 1},
    RUN_B: {UNDECLARED: 2, NO_MEMBER: 2},
}
PER_PROJECT = {PROJECT: OVERALL}
PER_ARM = {ARM: PER_RUN[RUN_A], ARM_B: PER_RUN[RUN_B]}
PER_ITEM = {ITEM_A: {UNDECLARED: 4, NO_MEMBER: 3}, ITEM_B: {NO_FILE: 2, TOO_MANY: 1}}
PER_ATTEMPT = {
    0: {UNDECLARED: 3, NO_MEMBER: 1, NO_FILE: 1},
    1: {UNDECLARED: 1, TOO_MANY: 1, NO_MEMBER: 1},
    2: {NO_FILE: 1, NO_MEMBER: 1},
}
# T2's NO_FILE comes at attempts 0 and 2, never at two attempts in a row, so it does not recur.
RECURS = {(RUN_A, T1.trial_id): True, (RUN_A, T2.trial_id): False, (RUN_B, T4.trial_id): True,
          (RUN_B, T5.trial_id): False}


def build_trial(store: TextStore, plan: Plan) -> Trial:
    """Return the plan's Trial: one attempt and one request per entry, every text held in `store`."""
    attempts: list[Attempt] = []
    requests: list[Request] = []
    for index, diagnostics in enumerate(plan.attempts):
        prompt = store.put(f"{PROMPT_MARKER} {index}\n// FILE: scale.cpp\n// {REFERENCE_MARKER}\nint main() {{}}\n")
        reply = f"{REPLY_MARKER} {index}: calls {IDENT_MARKER}\n"
        errors = any(item.stage == "compile" and item.severity == "error" for item in diagnostics)
        attempts.append(Attempt(
            index=index, prompt_ref=prompt, response_text=reply, files={"scale.cpp": f"// {FILES_MARKER} {index}\n"},
            stage_reached="S1" if errors else "S4", diagnostics=list(diagnostics),
        ))
        requests.append(Request(
            index=index, stage="generate" if index == 0 else "compile_loop", attempt_index=index,
            messages=[RequestMessage(role="user", ref=prompt)], reply_ref=store.put(reply),
        ))
    provenance = Provenance(
        commit=SYNTHETIC_COMMIT, dirty=False, device=None, sdk=None, date=PLACEHOLDER_DATE, device_records=[]
    )
    return Trial(
        trial_id=plan.trial_id,
        recipe_hash=SYNTHETIC_RECIPE_HASH,
        toolchain_pins=ToolchainPins(**dict(PLACEHOLDER_PINS)),
        provenance=provenance,
        bench_item=BenchItem(suite=SUITE, item=plan.item, split="eval", direction=plan.direction),
        model=ModelInfo(backend="replay", id=plan.arm, sampling=Sampling(0.2, 0.9, 64)),
        context=Context(knowledge_summary=CONTEXT_MARKER, source_description=CONTEXT_MARKER),
        requests=requests,
        attempts=attempts,
        final=Final(stage_reached=attempts[-1].stage_reached, corrections=len(attempts) - 1),
    )


def run_dir(root: Path, run_id: str) -> Path:
    """Return a run's directory under the runs root, as the runner places it."""
    return root / "runs" / run_id


def record_path(root: Path, plan: Plan) -> Path:
    """Return the trial.json of a plan's trial."""
    return trial_dir(run_dir(root, plan.run_id), plan.trial_id) / TRIAL_JSON


def build_tree(root: Path) -> Path:
    """Write every plan's trial under the runs root `root`, and a decoy trial.json in a build directory; return root.

    The decoy is a file a model could write among its files: no record, and
    not at a trial's depth, so the tool never reads it.
    """
    for plan in PLANS:
        directory = run_dir(root, plan.run_id)
        write_trial(build_trial(TextStore(directory), plan), directory, TextStore(directory))
    decoy = trial_dir(run_dir(root, RUN_A), T1.trial_id) / "attempt00" / "build" / TRIAL_JSON
    decoy.parent.mkdir(parents=True)
    decoy.write_bytes(b"SYNTHETIC: not a record\n")
    return root


def nested(mapping: Mapping[Any, Mapping[str, int]]) -> dict[Any, dict[str, int]]:
    """Return a mapping of mappings as plain dicts, leaving out an inner mapping that is empty."""
    return {key: dict(inner) for key, inner in mapping.items() if inner}


def snapshot(root: Path) -> dict[str, str]:
    """Return every path under `root` with the sha256 of its bytes, or 'dir' for a directory."""
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "dir"
        for path in sorted(root.rglob("*"))
    }


def forbidden_texts() -> set[str]:
    """Return every text the report must not hold: the markers, raw messages, quoted tokens, paths, and left-out ids."""
    texts = set(MARKERS) | {T3.trial_id, LINKER, *IGNORED_BUCKETS}
    for plan in PLANS:
        for diagnostics in plan.attempts:
            for item in diagnostics:
                texts.add(item.message)
                texts.update(re.findall(r"'[^']*'", item.message))
                texts.update(re.findall(r"[^\s'\":,;()]*/[^\s'\":,;()]*", item.message))
    return texts - {""}


def count_line(text: str, bucket: str, count: int) -> bool:
    """Return True when some line of `text` holds `bucket` and, apart from it, `count` as a number of its own."""
    return any(
        bucket in line and str(count) in re.split(r"[^0-9]+", line.replace(bucket, " "))
        for line in text.splitlines()
    )


def named_in(message: str, path: Path, root: Path) -> bool:
    """Return True when `message` names `path`, written natively, with / separators, or relative to `root`."""
    return any(form in message for form in (str(path), path.as_posix(), path.relative_to(root).as_posix()))


# ---------------------------------------------------------------------------
# classify: the counts and the recurrence flag


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """Return a runs root holding the SYNTHETIC trials of PLANS."""
    return build_tree(tmp_path / "runs-root")


def test_the_counts_per_bucket_overall_run_project_arm_item_and_attempt(tree: Path) -> None:
    found = tool().classify(tree)
    assert dict(found.overall) == OVERALL
    assert nested(found.per_run) == PER_RUN
    assert nested(found.per_project) == PER_PROJECT
    assert nested(found.per_arm) == PER_ARM
    assert nested(found.per_item) == PER_ITEM
    assert nested(found.per_attempt) == PER_ATTEMPT


def test_the_provenance_of_every_run_from_its_records(tree: Path) -> None:
    found = dict(tool().classify(tree).provenance)
    assert sorted(found) == [RUN_A, RUN_B]
    for run_id, stamps in found.items():
        seen = [(stamp.commit, stamp.dirty, stamp.device, stamp.sdk, stamp.date, stamp.pins) for stamp in stamps]
        assert seen == [(SYNTHETIC_COMMIT, False, None, None, PLACEHOLDER_DATE, PLACEHOLDER_PINS)], run_id


def test_the_recurrence_flag_of_every_kept_trial(tree: Path) -> None:
    assert dict(tool().classify(tree).recurs) == RECURS


def test_tt_to_cpp_trials_are_left_out(tree: Path) -> None:
    found = tool().classify(tree)
    assert LINKER not in found.overall, "the tt -> cpp trial's bucket is counted"
    assert all(trial_id != T3.trial_id for _, trial_id in found.recurs), "the tt -> cpp trial has a recurrence flag"
    # Without the cpp -> tt trials of run A, only the tt -> cpp trial is left there, so run A has no count.
    for plan in (T1, T2):
        record_path(tree, plan).unlink()
    found = tool().classify(tree)
    assert not dict(found.per_run).get(RUN_A), "a run whose only trial goes tt -> cpp holds counts"
    assert nested(found.per_run) == {RUN_B: PER_RUN[RUN_B]}
    assert nested(found.per_arm) == {ARM_B: PER_RUN[RUN_B]}, "an arm whose only trial goes tt -> cpp holds counts"


def test_only_compile_stage_errors_are_bucketed(tree: Path) -> None:
    found = tool().classify(tree)
    for bucket in IGNORED_BUCKETS:
        assert bucket not in found.overall, f"{bucket!r} is not a compile-stage error"


# ---------------------------------------------------------------------------
# Malformed records


def not_json(path: Path) -> None:
    """Replace the record with text that is not JSON."""
    path.write_bytes(b"{ SYNTHETIC: not JSON\n")


def without_provenance(path: Path) -> None:
    """Drop the record's provenance, which the strict loader requires."""
    data = json.loads(path.read_text(encoding="ascii"))
    del data["provenance"]
    path.write_text(json.dumps(data), encoding="ascii")


def bad_severity(path: Path) -> None:
    """Give one diagnostic a severity outside the record's list."""
    data = json.loads(path.read_text(encoding="ascii"))
    data["attempts"][0]["diagnostics"][0]["severity"] = "fatal"
    path.write_text(json.dumps(data), encoding="ascii")


BREAKS = {"not-json": not_json, "no-provenance": without_provenance, "bad-severity": bad_severity}


@pytest.mark.parametrize("kind", sorted(BREAKS))
def test_a_malformed_record_is_refused_naming_its_path(
    tree: Path, kind: str, capsys: pytest.CaptureFixture[str]
) -> None:
    path = record_path(tree, T2)
    BREAKS[kind](path)
    module = tool()
    with pytest.raises(ValueError) as caught:
        module.classify(tree)
    assert named_in(str(caught.value), path, tree), str(caught.value)
    capsys.readouterr()
    assert module.main([str(tree)]) == 2
    out, err = capsys.readouterr()
    assert named_in(err, path, tree), err
    assert out == "", "a refused record gives no report"


def test_a_decoy_in_a_build_directory_is_not_read(tree: Path) -> None:
    decoy = trial_dir(run_dir(tree, RUN_A), T1.trial_id) / "attempt00" / "build" / TRIAL_JSON
    assert decoy.is_file()
    assert dict(tool().classify(tree).overall) == OVERALL


# ---------------------------------------------------------------------------
# The command line: counts and patterns only; nothing written, nothing run


def check_report(text: str) -> None:
    """Check the report's counts, names, and provenance, and that it holds no forbidden text."""
    for bucket, count in OVERALL.items():
        assert count_line(text, bucket, count), f"no line holds {bucket!r} with its count {count}:\n{text}"
    names = (RUN_A, RUN_B, PROJECT, ARM, ARM_B, SYNTHETIC_COMMIT, PLACEHOLDER_DATE)
    names += tuple(pin for _, pin in PLACEHOLDER_PINS) + tuple(plan.trial_id for plan in KEPT)
    for name in names:
        assert name in text, f"the report does not name {name!r}"
    leaked = sorted(found for found in forbidden_texts() if found in text)
    assert leaked == [], f"the report holds record text: {leaked}"


def test_main_prints_counts_and_patterns_only(tree: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert tool().main([str(tree)]) == 0
    out, err = capsys.readouterr()
    check_report(out)
    leaked = sorted(found for found in forbidden_texts() if found in err)
    assert leaked == [], f"stderr holds record text: {leaked}"


def test_the_script_runs_from_the_command_line(tree: Path) -> None:
    assert TOOL.is_file(), f"{TOOL.relative_to(REPO).as_posix()} does not exist yet (task P17.24)"
    result = subprocess.run(
        [sys.executable, str(TOOL), str(tree)], cwd=REPO, capture_output=True, text=True, timeout=300, check=False
    )
    assert result.returncode == 0, result.stderr
    check_report(result.stdout)


def test_it_writes_nothing_and_starts_no_process(
    tmp_path: Path, tree: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = tool()
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the tool started a process")

    for name in ("Popen", "run", "call", "check_call", "check_output"):
        monkeypatch.setattr(subprocess, name, refuse)
    for name in ("system", "popen", "posix_spawn", "posix_spawnp", "execv", "execvp", "startfile"):
        monkeypatch.setattr(os, name, refuse, raising=False)
    before = snapshot(tmp_path)
    module.classify(tree)
    assert module.main([str(tree)]) == 0
    capsys.readouterr()
    assert snapshot(tmp_path) == before, "the tool wrote under the runs root or the working directory"
