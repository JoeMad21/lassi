"""Bucket the cpp -> tt build errors of run records by a fixed rule (task P17.24; bible Result Record).

Usage:

    python3 tools/classify_build_errors.py <runs root>

<runs root> holds runs/<run id>/, each a run tree as lassi.core.runner
writes it. The tool reads only the trial.json records at a trial's depth
under each run directory, one directory level per trial_id segment, so no
file in a build directory, the text store, or the binary store is read. It
writes nothing and starts no process. Besides the standard library it
imports only this repository's lassi.core.record, the strict record loader;
that module imports lassi.core.interfaces and lassi.harness.contract, and
they import only the standard library and lassi.core.capabilities and
lassi.harness.lassi_io, which import only the standard library. So python3
3.10 runs the tool from a checkout on the build host, and since it sets
sys.dont_write_bytecode before that import, it writes no __pycache__ there.

It keeps the trials whose bench_item.direction is cpp-tt, and in every
attempt of each it buckets every Diagnostic whose stage is compile and whose
severity is error; a warning, a note, and a diagnostic of another stage are
not bucketed. A diagnostic's bucket is its message abstracted by this fixed
rule, applied in this order (abstract_message):

1. Symbols. The rest of the message after the first "undefined symbol: "
   or "duplicate symbol: " (ld.lld) or "undefined reference to " (GNU ld),
   each space there standing for any run of whitespace, becomes <S> when it
   holds a character other than whitespace. A parsed diagnostic is one
   line, so that is the rest of the line. Then each remaining C++ qualified
   name, a maximal run of ASCII letters, digits, _, and :: that holds ::,
   and each identifier (an ASCII letter or _, then ASCII letters, digits,
   and _) that no letter, digit, or _ precedes and that ( directly follows,
   becomes <S>.
2. Quoted tokens. Each span that opens with ' and closes at the next ',
   opens with " and closes at the next ", opens with ` and closes at the
   next ' (the linker's quoting), or opens with U+2018 and closes at the
   next U+2019 (U+201C and U+201D likewise), the quotes included, becomes
   <Q>. A quote that never closes stays as it is.
3. Paths. Each maximal run of characters other than whitespace, the quote
   characters of step 2, and the characters : , ; ( ) that holds at least
   one /, or that ends in .cpp, .cc, .c, .hpp, .h, or .hh after at least
   one other character (a bare source file name), becomes <P>.
4. Numbers. Each maximal run of decimal digits, with an optional . and more
   digits, or 0x or 0X and hex digits, that no identifier character (an
   ASCII letter or digit, _, $, or any character outside ASCII) touches on
   either side, becomes <N>. So the digits of an identifier such as
   uint32_t or c_16 stay here, and a sign before a number stays.
5. Other tokens. Each whitespace-separated token, its leading and trailing
   punctuation (the characters , . ; : ( ) [ ] { } ' " ! ? and `) set
   aside, becomes <X>, its punctuation kept around it, unless what is left
   is a plain lowercase ASCII word ([a-z]+, or such words joined by -, as
   in out-of-line) or, the placeholders <Q>, <P>, <N>, <S>, and <X> aside,
   holds only ASCII punctuation other than _ and $ (as in <P>:<N>, -<N>,
   or >>>). So any token holding a letter, digit, _, $, or non-ASCII
   character outside the placeholders that is not a plain lowercase ASCII
   word is masked, whatever the message's wording, such as a symbol after
   a linker phrase step 1 does not know.
6. Each run of whitespace becomes one space, and both ends are stripped.

The rule is idempotent: a bucket is its own bucket. A bucket is plain
ASCII, and what can survive in it is the placeholders, plain lowercase
ASCII words, and tokens of ASCII punctuation alone (for example >>> or
<N>.<N>), with the punctuation step 5 sets aside around them. Known limit:
a plain lowercase ASCII word the model chose stays as written, such as an
identifier named helper that a message prints unquoted, or a word of a
text the model wrote in its code that the compiler prints back (a #error
line, a static_assert or error attribute's message).

The report counts those diagnostics overall, per run, per project and per
arm (the trial_id's first two segments, so the combination and the model
are covered whatever a run directory holds), per bench item, and per
attempt index, and says for every kept trial whether a bucket recurs across
corrections (some bucket of attempt k is also a bucket of attempt k + 1).
Per run it prints the provenance its records carry (Agent Rule 1): each
distinct combination of the commit, dirty flag, device, sdk, and date they
copy from the run's manifest and the toolchain pins they record
(Trial.toolchain_pins, the toolchains the run used, each pin by name).
Every count is copied from the records, never measured here (Agent Rule 1).
The report holds counts, bucket patterns, run ids, projects, arms, items,
attempt indexes, trial ids, and that provenance only: never a prompt, a
reply, a file the model wrote, a reference's text, the trial's context, a
diagnostic's raw message, or its file (Agent Rule 5), the known limit above
aside. It is plain ASCII: a character outside printable ASCII in a name or
a provenance value is printed as a backslash escape.

A record that is not UTF-8 JSON, that the strict record loader refuses, or
whose trial_id does not name its place raises ValueError naming its path,
and the command line then exits 2 with that message on stderr and prints
no report; nothing is skipped silently. The message keeps the loader's
reason without the value it refused, at most 240 characters of it; the
reason can still name a key of the record (such as the name of a file the
model wrote) and, for a malformed trial_id, the id itself, which the
loader names before any refused value.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import string
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.dont_write_bytecode = True  # no __pycache__ in the checkout the tool runs from

from lassi.core.record import Attempt, TextRef, Trial, from_dict, parse_trial_id  # noqa: E402

TRIAL_JSON = "trial.json"
# The direction whose trials are kept.
CPP_TT = "cpp-tt"
# A trial directory lies one level per trial_id segment below the run directory.
TRIAL_DEPTH = 6
# Step 1: the linkers' symbol phrases, the rest of the message after the first of them, and the C++ name shapes.
_SYMBOL_TAIL = re.compile(r"(undefined\s+symbol:|duplicate\s+symbol:|undefined\s+reference\s+to)\s+(?=\S).*", re.DOTALL)
_SYMBOL = re.compile(
    r"(?<![A-Za-z0-9_])(?:[A-Za-z0-9_]|::)*::(?:[A-Za-z0-9_]|::)*"  # a qualified name
    r"|(?<![A-Za-z0-9_])[A-Za-z_][A-Za-z0-9_]*(?=\()"  # an identifier that ( directly follows
)
# The typographic quotes (U+2018, U+2019, U+201C, U+201D), built from their code points so this file stays
# plain ASCII.
LSQ, RSQ, LDQ, RDQ = chr(0x2018), chr(0x2019), chr(0x201C), chr(0x201D)
# Each opening quote of step 2 and the character that closes its span, and every quote character of step 2.
QUOTE_CLOSERS = {"'": "'", '"': '"', "`": "'", LSQ: RSQ, LDQ: RDQ}
QUOTE_CHARS = "'\"`" + LSQ + RSQ + LDQ + RDQ
# Step 3: a maximal run of characters that may hold a path, and a bare source file name.
_PATH_RUN = re.compile("[^\\s" + re.escape(QUOTE_CHARS + ":,;()") + "]+")
_SOURCE_NAME = re.compile(r".+\.(?:cpp|cc|c|hpp|h|hh)")
# Step 4: a hex or decimal number; whether a letter, digit, or _ touches it is checked apart.
_NUMBER = re.compile(r"0[xX][0-9A-Fa-f]+|[0-9]+(?:\.[0-9]+)?")
# Step 5: the punctuation set aside around a token, a plain lowercase word, and every placeholder of the rule.
TOKEN_PUNCTUATION = ",.;:()[]{}'\"!?`"
_PLAIN_WORD = re.compile(r"[a-z]+(?:-[a-z]+)*")
_PLACEHOLDER = re.compile(r"<[QPNSX]>")
# The characters a surviving token may hold besides placeholders: ASCII punctuation, _ and $ left out.
_ASCII_PUNCTUATION = frozenset(string.punctuation) - {"_", "$"}
# The longest loader reason a refusal message keeps.
_REASON_LIMIT = 240
# The groupings of the counts, by Classification field name.
GROUPS = ("per_run", "per_project", "per_arm", "per_item", "per_attempt")


class Stamp(NamedTuple):
    """A trial's provenance: commit, dirty, device, sdk, and date from its run's manifest, and its toolchain pins.

    pins holds (name, pin) for each toolchain the trial records a pin for
    (Trial.toolchain_pins, the fields that are not None), in the record's
    field order.
    """

    commit: str | None
    dirty: bool | None
    device: str | None
    sdk: str | None
    date: str
    pins: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class Classification:
    """The buckets of the kept trials' compile-stage errors, counted several ways.

    overall maps bucket -> count. per_run (run id), per_project and per_arm
    (the trial_id's first two segments), per_item (the bench item), and
    per_attempt (the attempt index, an int) each map their key -> bucket ->
    count. A count is a number of diagnostics, and no inner mapping is
    empty. recurs maps (run id, trial_id) of every kept trial, and no other,
    to whether a bucket recurs across corrections. trials_read maps the id
    of every run directory under runs/ to the number of trial records read
    there, every direction included, and provenance maps it to the distinct
    Stamps of those records, in the order first read.
    """

    overall: dict[str, int]
    per_run: dict[str, dict[str, int]]
    per_project: dict[str, dict[str, int]]
    per_arm: dict[str, dict[str, int]]
    per_item: dict[str, dict[str, int]]
    per_attempt: dict[int, dict[str, int]]
    recurs: dict[tuple[str, str], bool]
    trials_read: dict[str, int]
    provenance: dict[str, list[Stamp]]


# ---------------------------------------------------------------------------
# The rule


def abstract_message(message: str) -> str:
    """Return a diagnostic message's bucket by the fixed rule (module docstring, steps 1 to 6)."""
    text = _SYMBOL_TAIL.sub(r"\1 <S>", message, count=1)
    text = _SYMBOL.sub("<S>", text)
    text = _quoted_tokens(text)
    text = _PATH_RUN.sub(_path, text)
    text = _NUMBER.sub(_number, text)
    # Steps 5 and 6 at once: each whitespace-separated token masked, then joined by one space.
    return " ".join(_masked(token) for token in text.split())


def _quoted_tokens(text: str) -> str:
    """Return `text` with each closed quoted span, its quotes included, replaced by <Q> (step 2)."""
    pieces: list[str] = []
    copied = position = 0
    unmatched: set[str] = set()  # closers with no occurrence left past the scan position
    while position < len(text):
        closer = QUOTE_CLOSERS.get(text[position])
        end = -1
        if closer is not None and closer not in unmatched:
            end = text.find(closer, position + 1)
            if end < 0:
                unmatched.add(closer)
        if end < 0:
            position += 1
            continue
        pieces += [text[copied:position], "<Q>"]
        copied = position = end + 1
    pieces.append(text[copied:])
    return "".join(pieces)


def _path(match: re.Match[str]) -> str:
    """Return <P> for a run that holds a / or is a bare source file name, and the run as it is otherwise (step 3)."""
    run = match.group()
    return "<P>" if "/" in run or _SOURCE_NAME.fullmatch(run) else run


def _touches(char: str) -> bool:
    """Return True for an identifier character: an ASCII letter or digit, _, $, or any character outside ASCII."""
    return not char.isascii() or char.isalnum() or char in "_$"


def _number(match: re.Match[str]) -> str:
    """Return <N> for a number no identifier character touches, and the number as it is otherwise (step 4)."""
    text, start, end = match.string, match.start(), match.end()
    if (start > 0 and _touches(text[start - 1])) or (end < len(text) and _touches(text[end])):
        return match.group()
    return "<N>"


def _masked(token: str) -> str:
    """Return a whitespace-free token as step 5 leaves it: as it is, or <X> inside its outer punctuation.

    It stays when what is left inside its outer punctuation is a plain
    lowercase ASCII word, or holds only ASCII punctuation (_ and $ left
    out) besides the placeholders.
    """
    core = token.strip(TOKEN_PUNCTUATION)
    if _PLAIN_WORD.fullmatch(core) or set(_PLACEHOLDER.sub("", core)) <= _ASCII_PUNCTUATION:
        return token
    lead = len(token) - len(token.lstrip(TOKEN_PUNCTUATION))
    trail = len(token.rstrip(TOKEN_PUNCTUATION))
    return token[:lead] + "<X>" + token[trail:]


# ---------------------------------------------------------------------------
# The records


def load_record(path: Path) -> Trial:
    """Return the Trial a trial.json holds, loaded strictly; raise ValueError naming `path` when it cannot be.

    Each attempt's response_text must be a text-store reference, as the
    runner writes it; the reference is checked and the text is never read,
    so the Trial's response texts are empty.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as error:
        raise ValueError(f"{path}: not UTF-8 text ({error.reason} at byte {error.start})") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: not JSON ({error.msg} at line {error.lineno} column {error.colno})") from None
    attempts = data.get("attempts", []) if isinstance(data, dict) else None
    if not isinstance(attempts, list) or not all(isinstance(item, dict) for item in attempts):
        raise ValueError(f"{path}: refused by the record loader: expected a trial object whose attempts are objects")
    try:
        for attempt_data in attempts:
            from_dict(TextRef, _text_ref(attempt_data.get("response_text")))
            attempt_data["response_text"] = ""
        return from_dict(Trial, data)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{path}: refused by the record loader: {_reason(error)}") from None


def _text_ref(value: Any) -> Mapping[str, Any]:
    """Return a response_text value when it is an object, as a text-store reference is; raise ValueError otherwise."""
    if not isinstance(value, dict):
        raise ValueError("Attempt.response_text must be a text-store reference")
    return value


def _reason(error: Exception) -> str:
    """Return the loader's reason without the value it refused, in printable ASCII and at most _REASON_LIMIT long."""
    reason = _printable(str(error).split(", got ", 1)[0])
    return reason if len(reason) <= _REASON_LIMIT else reason[: _REASON_LIMIT - 3] + "..."


def record_paths(runs_root: Path) -> list[tuple[str, Path | None]]:
    """Return (run id, trial.json path) for every record under runs_root/runs, in sorted order.

    A run directory that holds no record at a trial's depth gives one entry
    whose path is None, so every run is named. Raises ValueError when
    runs_root/runs is not a directory.
    """
    runs = Path(runs_root) / "runs"
    if not runs.is_dir():
        raise ValueError(f"{runs}: not a directory; expected <runs root>/runs/<run id>/")
    found: list[tuple[str, Path | None]] = []
    pattern = "/".join(["*"] * TRIAL_DEPTH + [TRIAL_JSON])
    for run_dir in sorted(entry for entry in runs.iterdir() if entry.is_dir()):
        paths = sorted(path for path in run_dir.glob(pattern) if path.is_file())
        found += [(run_dir.name, path) for path in paths] or [(run_dir.name, None)]
    return found


def _placed_trial(path: Path) -> Trial:
    """Load a record and check that its trial_id names its place below the run directory."""
    trial = load_record(path)
    place = "/".join(path.parent.parts[-TRIAL_DEPTH:])
    if trial.trial_id != place:
        raise ValueError(f"{path}: the record's trial_id does not name its place {place!r}")
    return trial


def attempt_buckets(attempt: Attempt) -> list[str]:
    """Return the bucket of every compile-stage error of an attempt, in record order."""
    return [
        abstract_message(item.message)
        for item in attempt.diagnostics
        if item.stage == "compile" and item.severity == "error"
    ]


def _stamp(trial: Trial) -> Stamp:
    """Return the Stamp of a trial: its provenance copy and the toolchain pins it records."""
    found = trial.provenance
    pins = tuple(
        (spec.name, getattr(trial.toolchain_pins, spec.name))
        for spec in dataclasses.fields(trial.toolchain_pins)
        if getattr(trial.toolchain_pins, spec.name) is not None
    )
    return Stamp(found.commit, found.dirty, found.device, found.sdk, found.date, pins)


# ---------------------------------------------------------------------------
# The classification


def classify(runs_root: str | Path) -> Classification:
    """Read every record under runs_root/runs and bucket the cpp -> tt compile-stage errors.

    See Classification for what each mapping holds. A record that cannot be
    loaded raises ValueError naming its path (module docstring).
    """
    overall: Counter[str] = Counter()
    groups: dict[str, dict[Any, Counter[str]]] = {name: {} for name in GROUPS}
    recurs: dict[tuple[str, str], bool] = {}
    trials_read: dict[str, int] = {}
    stamps: dict[str, dict[Stamp, None]] = {}
    for run_id, path in record_paths(Path(runs_root)):
        trials_read.setdefault(run_id, 0)
        stamps.setdefault(run_id, {})
        if path is None:
            continue
        trial = _placed_trial(path)
        trials_read[run_id] += 1
        stamps[run_id][_stamp(trial)] = None
        if trial.bench_item.direction == CPP_TT:
            recurs[(run_id, trial.trial_id)] = _count_trial(trial, run_id, overall, groups)
    plain = {name: {key: dict(inner) for key, inner in groups[name].items()} for name in GROUPS}
    return Classification(
        overall=dict(overall),
        **plain,
        recurs=recurs,
        trials_read=trials_read,
        provenance={run_id: list(found) for run_id, found in stamps.items()},
    )


def _count_trial(trial: Trial, run_id: str, overall: Counter[str], groups: dict[str, dict[Any, Counter[str]]]) -> bool:
    """Count a kept trial's buckets into `overall` and `groups`; return whether a bucket recurs across corrections."""
    parsed = parse_trial_id(trial.trial_id)
    buckets = [attempt_buckets(attempt) for attempt in trial.attempts]
    for attempt, found in zip(trial.attempts, buckets, strict=True):
        keys = (run_id, parsed.project, parsed.arm, trial.bench_item.item, attempt.index)
        for bucket in found:
            overall[bucket] += 1
            for name, key in zip(GROUPS, keys, strict=True):
                groups[name].setdefault(key, Counter())[bucket] += 1
    pairs = zip(buckets, buckets[1:], strict=False)  # attempt k and attempt k + 1
    return any(set(now) & set(after) for now, after in pairs)


# ---------------------------------------------------------------------------
# The report


def _printable(text: str) -> str:
    """Return `text` with every character outside printable ASCII written as a backslash escape."""
    out = []
    for char in text:
        code = ord(char)
        if 0x20 <= code < 0x7F:
            out.append(char)
        elif code < 0x100:
            out.append(f"\\x{code:02x}")
        elif code < 0x10000:
            out.append(f"\\u{code:04x}")
        else:
            out.append(f"\\U{code:08x}")
    return "".join(out)


def _count_lines(counts: Mapping[str, int], indent: str) -> list[str]:
    """Return one line per bucket, its count first, by count descending and then bucket."""
    ordered = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
    return [f"{indent}{count:>7}  {_printable(bucket) if bucket else '(empty message)'}" for bucket, count in ordered]


def _grouped(title: str, groups: Mapping[Any, Mapping[str, int]]) -> list[str]:
    """Return a titled section with each group's name and its bucket lines, groups in sorted order."""
    lines = ["", title]
    for key in sorted(groups):
        lines += [f"  {_printable(str(key))}", *_count_lines(groups[key], "  ")]
    return lines if groups else [*lines, "  (none)"]


def _known(value: object, unknown: str) -> str:
    """Return a provenance value as printable ASCII: true or false for a flag, and `unknown` for None."""
    if value is None:
        return unknown
    return ("true" if value else "false") if isinstance(value, bool) else _printable(str(value))


def _run_lines(result: Classification) -> list[str]:
    """Return the runs section: each run's tallies, then each distinct provenance its records copy."""
    lines = ["runs (records read, cpp -> tt trials kept, errors bucketed, kept trials with a recurring bucket)"]
    for run_id in sorted(result.trials_read):
        kept = [flag for (run, _), flag in result.recurs.items() if run == run_id]
        errors = sum(result.per_run.get(run_id, {}).values())
        lines.append(
            f"  {_printable(run_id)}: read {result.trials_read[run_id]}, kept {len(kept)}, "
            f"errors {errors}, recurring {sum(kept)}"
        )
        for stamp in result.provenance.get(run_id, []):
            lines.append(
                f"    provenance: commit {_known(stamp.commit, 'unknown')}, dirty {_known(stamp.dirty, 'unknown')}, "
                f"device {_known(stamp.device, 'none')}, sdk {_known(stamp.sdk, 'none')}, date {_known(stamp.date, '')}"
            )
            pins = ", ".join(f"{name} {_printable(pin)}" for name, pin in stamp.pins)
            lines.append(f"      toolchain pins: {pins or 'none recorded'}")
    return lines if result.trials_read else [*lines, "  (none)"]


def report(result: Classification) -> str:
    """Return the plain ASCII report of a Classification: counts, buckets, names, and the runs' provenance."""
    lines = [
        "cpp -> tt compile-stage errors by bucket, counted from trial.json records (tools/classify_build_errors.py)",
        "rule: symbols -> <S>, then quoted tokens -> <Q>, then paths -> <P>, then numbers -> <N>, "
        "then other tokens but plain lowercase words -> <X>, then whitespace collapsed",
        "",
        *_run_lines(result),
        "",
        f"overall (count, bucket; {sum(result.overall.values())} errors)",
    ]
    lines += _count_lines(result.overall, "  ") or ["  (none)"]
    lines += _grouped("per run", result.per_run)
    lines += _grouped("per project (the trial_id's first segment)", result.per_project)
    lines += _grouped("per arm (the trial_id's second segment, the model)", result.per_arm)
    lines += _grouped("per item", result.per_item)
    lines += _grouped("per attempt index (0 is the first generation, then one per correction)", result.per_attempt)
    lines += ["", "kept trials (run id, trial id, whether a bucket recurs across corrections)"]
    for (run_id, trial_id), flag in sorted(result.recurs.items()):
        lines.append(f"  {_printable(run_id)}  {_printable(trial_id)}  recurs {'yes' if flag else 'no'}")
    if not result.recurs:
        lines.append("  (none)")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line: print the report and return 0, or return 2 with a message on stderr for a refusal."""
    parser = argparse.ArgumentParser(
        prog="classify_build_errors.py",
        description="Bucket the cpp -> tt compile-stage errors of the trial.json records under a runs root.",
    )
    parser.add_argument("runs_root", type=Path, help="the runs root, which holds runs/<run id>/")
    args = parser.parse_args(argv)
    try:
        text = report(classify(args.runs_root))
    except (ValueError, OSError) as error:
        # The loader's reason is already printable ASCII; the path is printed as it is (stderr escapes the rest).
        print(f"classify_build_errors: refused: {error}", file=sys.stderr)
        return 2
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
