"""Render a Trial as trial.md, the page a person reads to follow the trial without tooling.

The page shows the trial's provenance, each prompt, each attempt's code, the
unified diff from the previous attempt, the parsed diagnostics, and the score
breakdown (bible Readability Standards, Trial row). It is plain ASCII with LF
newlines; any non-ASCII character is written as a Python backslash escape.
Every value that was not measured (None) shows as PLACEHOLDER. Provenance
values are not measurements, so an unknown one (None) shows as "-", as a
diagnostic without a code or location does.

The page is a list of blocks (headings, lines, tables, fenced code), each
ending with one newline and separated by one blank line. Fenced text goes
inside a fence longer than any backtick run it contains and is copied as is,
blank lines and trailing spaces included (a diff's context line for an empty
line is a single space), except that CRLF and lone CR line breaks become LF so
the page has LF newlines only. The exact texts stay in trial.json and the text
store.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from lassi.core.files import fence_for, language_for
from lassi.core.record import TOOLCHAIN_PIN_NAMES, Attempt, Context, Diagnostic, Provenance, TextRef, Trial

if TYPE_CHECKING:
    from lassi.core.store import TextStore

PLACEHOLDER = "PLACEHOLDER"


# ---------------------------------------------------------------------------
# Formatting pieces


def fmt(value: object) -> str:
    """Format one value for a table cell: None as PLACEHOLDER, bools in lowercase, floats by repr."""
    if value is None:
        return PLACEHOLDER
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, TextRef):
        return f"`{value.path}`"
    return str(value)


def fmt_provenance(value: object) -> str:
    """Format one provenance value: an unknown one (None) as '-', since provenance is not a measurement, else fmt."""
    return "-" if value is None else fmt(value)


def fenced(text: str, lang: str) -> str:
    """Return `text` in a fenced block one backtick longer than its longest backtick run.

    CRLF and lone CR line breaks become LF; everything else is kept as is.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    fence = fence_for(text)
    body = text if text.endswith("\n") else text + "\n"
    return f"{fence}{lang}\n{body}{fence}\n"


def _cell(text: str) -> str:
    """Escape a table cell: pipes become \\| and each line break becomes one space."""
    text = text.replace("|", "\\|")
    return text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Return a Markdown table block with escaped cells."""
    lines = [header, ["---"] * len(header), *rows]
    return "".join("| " + " | ".join(_cell(cell) for cell in line) + " |\n" for line in lines)


def _field_rows(record: object) -> list[tuple[str, str]]:
    """Return one (field name, formatted value) row per field of a record, in field order."""
    return [(spec.name, fmt(getattr(record, spec.name))) for spec in dataclasses.fields(record)]


# ---------------------------------------------------------------------------
# Page sections, each a list of blocks


def _summary_blocks(trial: Trial) -> list[str]:
    """Return the title and the table of trial-level fields."""
    bench, model, final = trial.bench_item, trial.model, trial.final
    sampling = model.sampling
    rows = [
        ("Recipe hash", f"`{trial.recipe_hash}`"),
        ("Suite", bench.suite),
        ("Item", bench.item),
        ("Direction", bench.direction),
        ("Split", bench.split),
        ("Model", f"{model.backend} `{model.id}`"),
        (
            "Sampling",
            f"temperature {fmt(sampling.temperature)}, top_p {fmt(sampling.top_p)}, "
            f"max_tokens {fmt(sampling.max_tokens)}",
        ),
        ("Stage reached", fmt(final.stage_reached)),
        ("Alignment", fmt(final.alignment)),
        ("Score", fmt(final.score)),
        ("Corrections", fmt(final.corrections)),
        ("Wall time (s)", fmt(final.wall_s)),
    ]
    return [f"# Trial {trial.trial_id}\n", _table(("Field", "Value"), rows)]


def _provenance_blocks(provenance: Provenance) -> list[str]:
    """Return the provenance table, one row per field in field order; an unknown value reads '-'."""
    rows = [(spec.name, fmt_provenance(getattr(provenance, spec.name))) for spec in dataclasses.fields(provenance)]
    return ["## Provenance\n", _table(("Field", "Value"), rows)]


def _pins_blocks(trial: Trial) -> list[str]:
    """Return the toolchain pins table; an unset pin reads 'not used'."""
    pins = [getattr(trial.toolchain_pins, name) for name in TOOLCHAIN_PIN_NAMES]
    rows = [(name, "not used" if pin is None else pin) for name, pin in zip(TOOLCHAIN_PIN_NAMES, pins, strict=True)]
    return ["## Toolchain pins\n", _table(("Toolchain", "Pin"), rows)]


def _context_blocks(context: Context) -> list[str]:
    """Return the knowledge summary and source description, each fenced, or None."""
    blocks = ["## Context\n"]
    sections = (("Knowledge summary", context.knowledge_summary), ("Source description", context.source_description))
    for title, text in sections:
        blocks += [f"### {title}\n", fenced(text, "text") if text else "None.\n"]
    return blocks


def _prompt_blocks(attempt: Attempt, store: TextStore) -> list[str]:
    """Return the prompt text, resolved through the store, with its reference."""
    ref = attempt.prompt_ref
    if ref is None:
        return ["### Prompt\n", "None.\n"]
    stored = f"Stored as `{ref.path}` (sha256 `{ref.sha256}`).\n"
    return ["### Prompt\n", stored, fenced(store.get(ref), "text")]


def _code_blocks(files: Mapping[str, str]) -> list[str]:
    """Return each file of the attempt, in sorted path order, fenced by its language."""
    if not files:
        return ["### Code\n", "None.\n"]
    blocks = ["### Code\n"]
    for path in sorted(files):
        blocks += [f"#### `{path}`\n", fenced(files[path], language_for(path))]
    return blocks


def _diff_blocks(attempt: Attempt) -> list[str]:
    """Return the unified diff from the previous attempt, or a line saying there is none."""
    if attempt.diff_from_previous:
        body = fenced(attempt.diff_from_previous, "diff")
    elif attempt.index == 0:
        body = "None (initial attempt).\n"
    else:
        body = "No changes.\n"
    return ["### Diff from previous attempt\n", body]


def _location(diagnostic: Diagnostic) -> str:
    """Return file, file:line, or file:line:column as far as set; '-' without a file."""
    if diagnostic.file is None:
        return "-"
    location = diagnostic.file
    if diagnostic.line is not None:
        location += f":{diagnostic.line}"
        if diagnostic.column is not None:
            location += f":{diagnostic.column}"
    return location


def _diagnostic_blocks(diagnostics: Sequence[Diagnostic]) -> list[str]:
    """Return the parsed diagnostics as a table, or None."""
    if not diagnostics:
        return ["### Diagnostics\n", "None.\n"]
    header = ("Stage", "Severity", "Code", "Location", "Message")
    rows = [(d.stage, d.severity, d.code or "-", _location(d), d.message) for d in diagnostics]
    return ["### Diagnostics\n", _table(header, rows)]


def _measurement_blocks(attempt: Attempt) -> list[str]:
    """Return the Run, Alignment, Profile, Guards, and Score breakdown tables."""
    alignment, score = attempt.alignment, attempt.score
    per_input = ", ".join(fmt(value) for value in alignment.per_input) or PLACEHOLDER
    alignment_rows = [("per_input", per_input), ("mean", fmt(alignment.mean))]
    score_rows = [(name, fmt(score.components[name])) for name in sorted(score.components)]
    score_rows.append(("scalar", fmt(score.scalar)))
    fields = ("Field", "Value")
    return [
        "### Run\n",
        _table(fields, _field_rows(attempt.run)),
        "### Alignment\n",
        _table(fields, alignment_rows),
        "### Profile\n",
        _table(fields, _field_rows(attempt.profile)),
        "### Guards\n",
        _table(fields, _field_rows(attempt.guards)),
        "### Score breakdown\n",
        _table(("Component", "Value"), score_rows),
    ]


def _attempt_blocks(attempt: Attempt, store: TextStore) -> list[str]:
    """Return the whole section of one attempt."""
    blocks = [f"## Attempt {attempt.index}\n", f"Stage reached: {attempt.stage_reached}\n"]
    blocks += _prompt_blocks(attempt, store)
    blocks += _code_blocks(attempt.files)
    blocks += _diff_blocks(attempt)
    blocks += _diagnostic_blocks(attempt.diagnostics)
    blocks += _measurement_blocks(attempt)
    return blocks


# ---------------------------------------------------------------------------
# The page


def render_trial_md(trial: Trial, store: TextStore) -> str:
    """Return trial.md for `trial`, resolving prompt texts through `store`.

    The result is deterministic, plain ASCII (non-ASCII characters become
    backslash escapes), uses LF newlines, and ends with exactly one newline.
    """
    blocks = _summary_blocks(trial) + _provenance_blocks(trial.provenance)
    blocks += _pins_blocks(trial) + _context_blocks(trial.context)
    for attempt in trial.attempts:
        blocks += _attempt_blocks(attempt, store)
    page = "\n".join(blocks)
    return page.encode("ascii", "backslashreplace").decode("ascii")
