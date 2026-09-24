"""Prompts built from a fragment prompt set, and the one-fenced-block reply form (bible Design Principles 3 and 4).

A fragment prompt set (lassi.prompts.assets: a manifest tree such as the
generated lassi-2024 set) holds checked text fragments rather than
templates. The stages join them in a fixed order; joiners that are only
whitespace live here, never in the fragments. The keys below are the
contract between such a set and the stages: a set that provides them works
with summarize_context, describe_source, and generate. A key template holds
`{SOURCE}` and `{TARGET}` (the direction's languages, upper case) or
`{target}` (the target language as written in the recipe); fragment_key
fills them.

Context packs are one-entry manifest trees; a pack serves the language its
entry key ends in (`<name>.<language>`), and pack_language reads it.

The functions here are pure text functions. The toggles that select
upstream's quirks (lassi.core.recipe.FIXES) are read by the stages:

- collapse_spaces is the `prompt_spaces` quirk: before its first generation
  call upstream cuts every run of spaces in the assembled prompt to one
  space; tabs and newlines stay.
- first_fence is the `fence_tag` quirk: the first fenced block with
  upstream's tag stripping (a leading `cpp` or `c++` is cut, then one more
  leading `c`). A cut that leaves text on the block's first line is a
  fence-quirk hit: a `cuda` tag leaves `uda`. With the fix on, generate
  reads FILE blocks instead (lassi.core.files).
- as_text_mode gives a source as a file opened in text mode reads it
  (universal newlines), which is how upstream reads its sources.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from lassi.bench import Direction

# The general system prompt, sent with the context summary and the source description requests.
GENERAL_SYSTEM = "system_prompt_dict.general_system"
# The direction's system prompt and translation request, sent with the generation prompt.
DIRECTION_SYSTEM = "system_prompt_dict.{SOURCE}_to_{TARGET}"
DIRECTION_REQUEST = "codetranslate_prompt_dict.{SOURCE}_to_{TARGET}"

# The context summary request: intro, an optional note for the target language, outro, then the pack.
SUMMARY_INTRO = "summarize_context.intro"
SUMMARY_NOTE = "summarize_context.{target}_note"
SUMMARY_OUTRO = "summarize_context.outro"
# The source description request: intro, then the source.
DESCRIPTION_INTRO = "describe_source.intro"

# The generation prompt's parts, in the order they are joined.
CONTEXT_OPEN = "generate.context_open"
CONTEXT_CLOSE = "generate.context_close"
SUMMARY_OPEN = "generate.summary_open"
SUMMARY_CLOSE = "generate.summary_close"
CONTEXT_LEAD = "generate.context_lead"
DESCRIPTION_LEAD = "generate.description_lead"
NO_CONTEXT_LEAD = "generate.no_context_lead"
REQUEST_LEAD = "generate.request_lead"

# The fragment keys each stage needs, as key templates (fragment_key fills them per direction).
SUMMARY_KEYS = (GENERAL_SYSTEM, SUMMARY_INTRO, SUMMARY_OUTRO)
DESCRIPTION_KEYS = (GENERAL_SYSTEM, DESCRIPTION_INTRO)
GENERATE_KEYS = (
    GENERAL_SYSTEM,
    DIRECTION_SYSTEM,
    DIRECTION_REQUEST,
    CONTEXT_OPEN,
    CONTEXT_CLOSE,
    SUMMARY_OPEN,
    SUMMARY_CLOSE,
    CONTEXT_LEAD,
    DESCRIPTION_LEAD,
    NO_CONTEXT_LEAD,
    REQUEST_LEAD,
)

# The Diagnostic codes of the one-fenced-block reply form.
FENCE_QUIRK = "fence-quirk"
NO_FENCE = "no-fence"

# The first fenced block: the shortest text between two runs of three backticks, across lines.
_FENCE = re.compile(r"```(.*?)```", re.DOTALL)
# Runs of spaces (only U+0020; tabs and newlines are not spaces here).
_SPACES = re.compile(" +")
# How much of a block's first line a fence-quirk message quotes.
_QUOTE_CHARS = 40


def fragment_key(template: str, direction: Direction) -> str:
    """Return the fragment key `template` names for `direction`."""
    return template.format(
        SOURCE=direction.source.upper(),
        TARGET=direction.target.upper(),
        target=direction.target,
    )


def pack_language(entry_key: str) -> str:
    """Return the language a context pack serves: the last dot-separated part of its one entry key."""
    return entry_key.rsplit(".", 1)[-1]


def collapse_spaces(text: str) -> str:
    """Return `text` with every run of spaces cut to one space; tabs and newlines stay."""
    return _SPACES.sub(" ", text)


def as_text_mode(text: str) -> str:
    """Return `text` as a file opened in text mode reads it: each CRLF and each lone CR becomes LF."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def summary_request(fragments: Mapping[str, str], direction: Direction, pack: str) -> str:
    """Return the context summary request: intro, the target's note when the set has one, outro, then the pack."""
    note = fragments.get(fragment_key(SUMMARY_NOTE, direction), "")
    return fragments[SUMMARY_INTRO] + note + fragments[SUMMARY_OUTRO] + pack


def description_request(fragments: Mapping[str, str], source: str) -> str:
    """Return the source description request: the intro, then the source text."""
    return fragments[DESCRIPTION_INTRO] + source


def generation_prompt(
    fragments: Mapping[str, str],
    direction: Direction,
    source: str,
    pack: str | None,
    summary: str,
    description: str,
) -> str:
    """Return the generation prompt, before any space collapse.

    With a pack: the pack between the context markers, the summary between
    the summary markers, the context lead, the description lead and the
    description. Without one: the general system prompt and the no-context
    lead. Then, in both cases, the request lead, the direction's translation
    request, one space, and the source.
    """
    if pack is None:
        head = fragments[GENERAL_SYSTEM] + fragments[NO_CONTEXT_LEAD]
    else:
        head = (
            fragments[CONTEXT_OPEN] + pack + fragments[CONTEXT_CLOSE]
            + fragments[SUMMARY_OPEN] + summary + fragments[SUMMARY_CLOSE]
            + fragments[CONTEXT_LEAD]
            + fragments[DESCRIPTION_LEAD] + description
        )
    request = fragments[fragment_key(DIRECTION_REQUEST, direction)]
    return head + fragments[REQUEST_LEAD] + request + " " + source


@dataclass(frozen=True)
class Fence:
    """The first fenced block of a reply.

    `found` is False when the reply holds no block, and `text` is then "".
    `removed` is what the tag stripping cut from the start of the block, and
    `quirk` says that cut left text on the block's first line (a fence-quirk
    hit).
    """

    found: bool
    text: str
    removed: str = ""
    quirk: bool = False


def first_fence(reply: str) -> Fence:
    """Return the first fenced block of `reply` after upstream's tag stripping.

    The block is the text between the first two runs of three backticks. A
    leading `c++` or `cpp` is cut, then one more leading `c`, whatever
    follows, as upstream does; any other tag stays in the text.
    """
    match = _FENCE.search(reply)
    if match is None:
        return Fence(found=False, text="")
    block = match.group(1)
    text = block[3:] if block.startswith(("c++", "cpp")) else block
    text = text[1:] if text.startswith("c") else text
    removed = block[: len(block) - len(text)]
    quirk = bool(removed) and bool(text.split("\n", 1)[0].strip())
    return Fence(found=True, text=text, removed=removed, quirk=quirk)


def fence_quirk_message(fence: Fence) -> str:
    """Return the message of a fence-quirk Diagnostic: what the stripping cut and what it left, in ASCII."""
    left = fence.text.split("\n", 1)[0][:_QUOTE_CHARS]
    return (
        f"upstream's fence tag stripping cut {ascii(fence.removed)} from the first fenced block, leaving "
        f"{ascii(left)} on its first line; the fence_tag fix strips only an exact tag"
    )
