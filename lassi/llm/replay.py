"""The replay LLM backend, registered as LLMBackend "replay" (bible Build Roadmap, P1 row; Design Principle 4).

It returns scripted, synthetic completions from a recording, one per call, in
the order they were recorded, and raises ServingError when they run out, so a
pipeline runs on fixed replies without a model host. A recording is written by
hand for a test: it is never model output, and nothing it yields is a
measurement.

A recording is a plain-ASCII JSON file labeled synthetic:

    {"synthetic": true,
     "completions": [{"text": "...", "messages": [{"role": "...", "content": "..."}]},
                     {"text": "..."}]}

Each entry holds the reply `text` and, optionally, the `messages` the call is
expected to send. When they are present, complete() compares the messages it
receives with them exactly (roles, order, count, and every character) and
raises ServingError on any difference. The file is read once, when the backend
is built. A file that is not ASCII, not valid JSON, not labeled
`"synthetic": true`, or not of this shape (an unknown key included, so a
misspelled `messages` cannot turn the check off) raises ValueError naming the
file; a file that cannot be read raises OSError. An empty completions list is
valid: a trial that ends before any model call needs no completion.

Its token counts are approximations, never measurements: whitespace word
counts (len(text.split())) of the messages and of the reply, as the mock's.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lassi.core.interfaces import Completion, Message, Sampling
from lassi.core.registry import register
from lassi.llm._http import ServingError

_TOP_KEYS = frozenset({"synthetic", "completions"})
_ENTRY_KEYS = frozenset({"text", "messages"})
_MESSAGE_KEYS = frozenset({"role", "content"})


@dataclass(frozen=True)
class _Recorded:
    """One recorded completion: the reply text and the (role, content) pairs expected, or None for no check."""

    text: str
    messages: tuple[tuple[str, str], ...] | None


@register("LLMBackend", "replay")
class ReplayBackend:
    """Replays a recording of scripted, synthetic completions in order; token counts are approximate word counts."""

    name = "replay"
    capabilities = frozenset({"chat"})

    def __init__(self, model_id: str, *, recording: str | os.PathLike[str] | None = None) -> None:
        """Keep `model_id` and read the recording at `recording`, a path to a synthetic JSON recording.

        The file is read and checked here, so a bad recording fails when the
        backend is built: ValueError naming the file for a bad one, OSError for
        one that cannot be read. With no recording (None), complete() raises
        ServingError.
        """
        self.model_id = model_id
        self._path = None if recording is None else Path(recording)
        self._completions = None if self._path is None else _load_recording(self._path)
        self._next = 0

    def __repr__(self) -> str:
        """Show the class and model_id only."""
        return f"{type(self).__name__}(model_id={self.model_id!r})"

    def complete(self, messages: Sequence[Message], sampling: Sampling) -> Completion:
        """Return the next recorded completion; `sampling` never changes it.

        When the entry records the messages it expects, `messages` must equal
        them exactly, or ServingError names the file and the first difference.
        ServingError also comes when the backend has no recording and, on this
        and every later call, when the recorded completions are used up. The
        token counts are deterministic approximations, never measurements:
        prompt_tokens is the sum of len(content.split()) over the messages and
        completion_tokens is len(text.split()) of the reply.
        """
        if self._completions is None:
            raise ServingError(f"{self!r} has no recording; build it with recording=<path to a synthetic JSON file>")
        if self._next >= len(self._completions):
            count = len(self._completions)
            raise ServingError(
                f"replay recording {self._path} is used up: it holds {count} completion(s) and all were returned"
            )
        recorded = self._completions[self._next]
        sent = tuple((message.role, message.content) for message in messages)
        if recorded.messages is not None and sent != recorded.messages:
            raise ServingError(
                f"replay recording {self._path}: call {self._next + 1} sent messages that differ from the "
                f"recorded ones: {_first_difference(recorded.messages, sent)}"
            )
        self._next += 1
        prompt_tokens = sum(len(content.split()) for _, content in sent)
        return Completion(text=recorded.text, prompt_tokens=prompt_tokens, completion_tokens=len(recorded.text.split()))


def _first_difference(expected: tuple[tuple[str, str], ...], sent: tuple[tuple[str, str], ...]) -> str:
    """Describe the first difference between two message lists by position and role, quoting no content."""
    for index, ((want_role, want), (got_role, got)) in enumerate(zip(expected, sent, strict=False), start=1):
        if want_role != got_role:
            return f"message {index} has role {got_role!r}, expected {want_role!r}"
        if want != got:
            shared = min(len(want), len(got))
            where = next((i for i in range(shared) if want[i] != got[i]), shared)
            lengths = f"length {len(got)}, expected {len(want)}"
            return f"message {index} ({want_role}) differs at character {where} ({lengths})"
    return f"{len(sent)} message(s) sent, {len(expected)} expected"


def _load_recording(path: Path) -> tuple[_Recorded, ...]:
    """Read and check the recording at `path`; raise ValueError naming the file when it is bad."""
    raw = path.read_bytes()
    if not raw.isascii():
        raise ValueError(f"replay recording {path} is not plain ASCII")
    try:
        data = json.loads(raw.decode("ascii"))
    except json.JSONDecodeError as error:
        raise ValueError(f"replay recording {path} is not valid JSON: {error.msg} at line {error.lineno}") from None
    if not isinstance(data, dict) or set(data) != _TOP_KEYS:
        raise ValueError(
            f'replay recording {path} must be an object with exactly the keys "synthetic" and "completions"'
        )
    if data["synthetic"] is not True:
        raise ValueError(f'replay recording {path} must be labeled "synthetic": true')
    if not isinstance(data["completions"], list):
        raise ValueError(f'replay recording {path}: "completions" must be a list')
    return tuple(_recorded(path, index, item) for index, item in enumerate(data["completions"], start=1))


def _recorded(path: Path, index: int, item: Any) -> _Recorded:
    """Return completion `index` of the recording at `path`, or raise ValueError naming the file and the entry."""
    where = f"replay recording {path}, completion {index}"
    if not isinstance(item, dict) or not set(item) <= _ENTRY_KEYS:
        raise ValueError(f'{where} must be an object with only the keys "text" and "messages"')
    if not isinstance(item.get("text"), str):
        raise ValueError(f'{where} must have a string "text"')
    if "messages" not in item:
        return _Recorded(text=item["text"], messages=None)
    messages = item["messages"]
    if not isinstance(messages, list):
        raise ValueError(f'{where}: "messages" must be a list')
    pairs = []
    for number, message in enumerate(messages, start=1):
        if not isinstance(message, dict) or set(message) != _MESSAGE_KEYS:
            raise ValueError(f'{where}, message {number} must be an object with exactly the keys "role" and "content"')
        if not isinstance(message["role"], str) or not isinstance(message["content"], str):
            raise ValueError(f'{where}, message {number}: "role" and "content" must be strings')
        pairs.append((message["role"], message["content"]))
    return _Recorded(text=item["text"], messages=tuple(pairs))
