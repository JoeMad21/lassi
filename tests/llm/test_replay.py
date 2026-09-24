"""Tests for the replay LLM backend in lassi/llm/replay.py (P1.9, first acceptance bullet).

The replay backend is registered as LLMBackend "replay", like the mock,
openai_compat, and ollama backends (bible Component Interfaces, LLMBackend
row). It returns recorded completions in the order they were recorded and
fails with a clear ServingError when they run out.

Its recording is a plain-ASCII JSON file (the package uses only the standard
library) holding scripted, synthetic completions, labeled as such:

    {"synthetic": true,
     "completions": [{"text": "...", "messages": [{"role": "...", "content": "..."}]},
                     {"text": "..."}]}

Each entry has a `text` and, optionally, the `messages` it expects; when they
are present, complete() checks the messages it receives against them exactly
and fails on any difference. A recording that is not ASCII, not labeled
`"synthetic": true`, or not of this shape is refused with ValueError when the
backend is built. An empty completions list is valid: a trial that ends before
any model call (a baseline failure) needs no completion.

Token counts follow the mock's rule (whitespace word counts), approximations
and never measurements. Every recording here is written into tmp_path by the
test itself; every text in it is made up for these tests and copies no
upstream source or prompt. No value in this module is a measurement.
"""

from __future__ import annotations

import importlib
import inspect
import json
import re
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from lassi import llm
from lassi.core.capabilities import Component
from lassi.core.interfaces import Completion, LLMBackend, Message, Sampling
from lassi.core.registry import DEFAULT_REGISTRY

ServingError = llm.ServingError

SAMPLING = Sampling(temperature=0.2, top_p=0.9, max_tokens=1024)
OTHER_SAMPLING = Sampling(temperature=1.3, top_p=0.1, max_tokens=7)

# Made-up fixture texts; none is model output or upstream text.
FIRST_TEXT = "```\nint fixture_first() { return 1; }\n```\n"
SECOND_TEXT = "Fixture reply two, no fence."
THIRD_TEXT = "```cpp\nint fixture_third() { return 3; }\n```\n"

MESSAGES = (
    Message(role="system", content="Fixture system text."),
    Message(role="user", content="Fixture user text  with two spaces."),
)
OTHER_MESSAGES = (Message(role="user", content="Some other fixture request."),)

# A recording written by hand, to pin the file format itself.
LITERAL_RECORDING = """\
{
  "synthetic": true,
  "completions": [
    {
      "messages": [
        {"role": "system", "content": "Fixture system text."},
        {"role": "user", "content": "Fixture user text  with two spaces."}
      ],
      "text": "```\\nint fixture_first() { return 1; }\\n```\\n"
    },
    {"text": "Fixture reply two, no fence."}
  ]
}
"""


def replay_module() -> Any:
    """Import lassi.llm.replay; a missing module fails the calling test, not the collection."""
    return importlib.import_module("lassi.llm.replay")


def replay_cls() -> Any:
    """Return the ReplayBackend class."""
    return replay_module().ReplayBackend


def entry(text: str, messages: tuple[Message, ...] | None = None) -> dict[str, Any]:
    """Return one recorded completion, with the expected messages when given."""
    data: dict[str, Any] = {"text": text}
    if messages is not None:
        data["messages"] = [{"role": m.role, "content": m.content} for m in messages]
    return data


def write_recording(path: Path, completions: list[dict[str, Any]]) -> Path:
    """Write a recording labeled synthetic, as ASCII JSON, and return its path."""
    data = {"synthetic": True, "completions": completions}
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="ascii")
    return path


def three_replies(tmp_path: Path) -> Path:
    """Write a recording of three completions with no expected messages."""
    return write_recording(tmp_path / "three.json", [entry(FIRST_TEXT), entry(SECOND_TEXT), entry(THIRD_TEXT)])


# ---------------------------------------------------------------------------
# Registration and shape


def test_registered_as_llm_backend_replay() -> None:
    cls = replay_cls()
    registry_entry = DEFAULT_REGISTRY.get("LLMBackend", "replay")
    assert registry_entry.factory is cls
    assert registry_entry.capabilities == cls.capabilities


def test_importing_the_package_registers_replay() -> None:
    # Registered like the other backends: importing lassi.llm alone registers it. A fresh interpreter,
    # because importing lassi.llm.replay in this process would register it whatever the package does.
    code = (
        "import sys, lassi.llm\n"
        "from lassi.core.registry import DEFAULT_REGISTRY\n"
        "print('lassi.llm.replay' in sys.modules, 'replay' in DEFAULT_REGISTRY.names('LLMBackend'))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["True", "True"], result.stdout


def test_name_and_capabilities() -> None:
    cls = replay_cls()
    assert cls.name == "replay"
    assert "name" in vars(cls) and "capabilities" in vars(cls)
    assert isinstance(cls.capabilities, frozenset)
    assert all(isinstance(capability, str) for capability in cls.capabilities)
    assert "chat" in cls.capabilities
    assert "needs_reference" not in cls.capabilities


def test_has_the_llm_backend_shape(tmp_path: Path) -> None:
    cls = replay_cls()
    assert inspect.getdoc(cls)
    params = list(inspect.signature(cls.complete).parameters)
    assert params == [*inspect.signature(LLMBackend.complete).parameters], params
    backend = cls("fixture/replay-a", recording=three_replies(tmp_path))
    assert isinstance(backend, Component)
    assert backend.model_id == "fixture/replay-a" and "model_id" in vars(backend)


def test_constructor_signature() -> None:
    params = list(inspect.signature(replay_cls()).parameters.values())
    assert [(p.name, p.kind) for p in params] == [
        ("model_id", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        ("recording", inspect.Parameter.KEYWORD_ONLY),
    ]
    assert params[1].default is None


def test_registry_factory_builds_a_backend_from_the_model_id_alone() -> None:
    # The construction convention: the runner builds an LLM backend as factory(model_id).
    backend = DEFAULT_REGISTRY.get("LLMBackend", "replay").factory("fixture/model-z")
    assert type(backend) is replay_cls()
    assert backend.model_id == "fixture/model-z"


def test_no_recording_raises_serving_error_naming_the_recording() -> None:
    with pytest.raises(ServingError) as info:
        replay_cls()("fixture/model-z").complete(MESSAGES, SAMPLING)
    assert "recording" in str(info.value)


def test_repr_shows_class_and_model_id_but_no_completion_text(tmp_path: Path) -> None:
    text = repr(replay_cls()("fixture/replay-a", recording=three_replies(tmp_path)))
    assert "ReplayBackend" in text and "fixture/replay-a" in text, text
    assert "fixture_first" not in text and "Fixture reply two" not in text, text


def test_model_info_names_the_replay_backend(tmp_path: Path) -> None:
    backend = replay_cls()("fixture/replay-a", recording=three_replies(tmp_path))
    info = llm.model_info(backend, SAMPLING)
    assert (info.backend, info.id, info.sampling) == ("replay", "fixture/replay-a", SAMPLING)


# ---------------------------------------------------------------------------
# Replay in order, and running out


def test_returns_recorded_completions_in_order(tmp_path: Path) -> None:
    backend = replay_cls()("fixture/replay-a", recording=three_replies(tmp_path))
    replies = [backend.complete(MESSAGES, SAMPLING) for _ in range(3)]
    assert all(isinstance(reply, Completion) for reply in replies)
    assert [reply.text for reply in replies] == [FIRST_TEXT, SECOND_TEXT, THIRD_TEXT]


def test_messages_and_sampling_do_not_change_the_replies(tmp_path: Path) -> None:
    backend = replay_cls()("fixture/replay-a", recording=three_replies(tmp_path))
    assert backend.complete(OTHER_MESSAGES, OTHER_SAMPLING).text == FIRST_TEXT
    assert backend.complete([], SAMPLING).text == SECOND_TEXT


def test_the_recording_path_may_be_a_str_or_a_path(tmp_path: Path) -> None:
    path = three_replies(tmp_path)
    for recording in (path, str(path)):
        assert replay_cls()("fixture/replay-a", recording=recording).complete(MESSAGES, SAMPLING).text == FIRST_TEXT


def test_fails_with_a_clear_error_when_the_completions_run_out(tmp_path: Path) -> None:
    path = write_recording(tmp_path / "exhausted.json", [entry(FIRST_TEXT), entry(SECOND_TEXT)])
    backend = replay_cls()("fixture/replay-a", recording=path)
    backend.complete(MESSAGES, SAMPLING)
    backend.complete(MESSAGES, SAMPLING)
    with pytest.raises(ServingError) as info:
        backend.complete(MESSAGES, SAMPLING)
    message = str(info.value)
    assert "exhausted.json" in message, message
    assert re.search(r"\b2\b", message), f"the error should give the number of recorded completions: {message}"
    # Running out is final: the backend never starts over.
    with pytest.raises(ServingError):
        backend.complete(MESSAGES, SAMPLING)


def test_an_empty_recording_loads_and_fails_on_the_first_call(tmp_path: Path) -> None:
    # A scenario that ends before any model call (a baseline failure) records no completion.
    backend = replay_cls()("fixture/replay-a", recording=write_recording(tmp_path / "empty.json", []))
    with pytest.raises(ServingError) as info:
        backend.complete(MESSAGES, SAMPLING)
    assert "empty.json" in str(info.value)


def test_each_backend_replays_its_recording_from_the_start(tmp_path: Path) -> None:
    path = three_replies(tmp_path)
    first = replay_cls()("fixture/replay-a", recording=path)
    assert first.complete(MESSAGES, SAMPLING).text == FIRST_TEXT
    second = replay_cls()("fixture/replay-a", recording=path)
    assert second.complete(MESSAGES, SAMPLING).text == FIRST_TEXT
    assert first.complete(MESSAGES, SAMPLING).text == SECOND_TEXT


def test_the_recording_is_read_once_when_the_backend_is_built(tmp_path: Path) -> None:
    path = three_replies(tmp_path)
    backend = replay_cls()("fixture/replay-a", recording=path)
    write_recording(path, [entry("Fixture text written later.")])
    assert backend.complete(MESSAGES, SAMPLING).text == FIRST_TEXT


def test_replay_starts_no_subprocess_and_opens_no_socket(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the replay backend started a subprocess or opened a socket")

    path = three_replies(tmp_path)
    cls = replay_cls()
    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(subprocess, "Popen", refuse)
    backend = cls("fixture/replay-a", recording=path)
    assert [backend.complete(MESSAGES, SAMPLING).text for _ in range(3)] == [FIRST_TEXT, SECOND_TEXT, THIRD_TEXT]


# ---------------------------------------------------------------------------
# Expected messages, checked when recorded


def test_expected_messages_that_match_pass(tmp_path: Path) -> None:
    path = write_recording(tmp_path / "checked.json", [entry(FIRST_TEXT, MESSAGES), entry(SECOND_TEXT, OTHER_MESSAGES)])
    backend = replay_cls()("fixture/replay-a", recording=path)
    assert backend.complete(list(MESSAGES), SAMPLING).text == FIRST_TEXT
    assert backend.complete(OTHER_MESSAGES, OTHER_SAMPLING).text == SECOND_TEXT


MISMATCHES = {
    "one-space-collapsed": (MESSAGES[0], Message("user", "Fixture user text with two spaces.")),
    "trailing-newline": (MESSAGES[0], Message("user", MESSAGES[1].content + "\n")),
    "role-differs": (Message("user", MESSAGES[0].content), MESSAGES[1]),
    "one-message-missing": (MESSAGES[0],),
    "one-message-extra": (*MESSAGES, Message("assistant", "Fixture extra text.")),
    "order-swapped": (MESSAGES[1], MESSAGES[0]),
}


@pytest.mark.parametrize("sent", MISMATCHES.values(), ids=MISMATCHES.keys())
def test_messages_that_differ_from_the_expected_ones_fail(tmp_path: Path, sent: tuple[Message, ...]) -> None:
    # The check is exact: roles, order, count, and every character of every content.
    path = write_recording(tmp_path / "mismatch.json", [entry(FIRST_TEXT, MESSAGES)])
    backend = replay_cls()("fixture/replay-a", recording=path)
    with pytest.raises(ServingError) as info:
        backend.complete(sent, SAMPLING)
    assert "mismatch.json" in str(info.value)


def test_only_entries_with_expected_messages_are_checked(tmp_path: Path) -> None:
    path = write_recording(tmp_path / "mixed.json", [entry(FIRST_TEXT), entry(SECOND_TEXT, MESSAGES)])
    backend = replay_cls()("fixture/replay-a", recording=path)
    assert backend.complete(OTHER_MESSAGES, SAMPLING).text == FIRST_TEXT
    with pytest.raises(ServingError):
        backend.complete(OTHER_MESSAGES, SAMPLING)


def test_the_hand_written_format_loads(tmp_path: Path) -> None:
    path = tmp_path / "literal.json"
    path.write_bytes(LITERAL_RECORDING.encode("ascii"))
    backend = replay_cls()("fixture/replay-a", recording=path)
    assert backend.complete(MESSAGES, SAMPLING).text == FIRST_TEXT
    assert backend.complete(OTHER_MESSAGES, SAMPLING).text == SECOND_TEXT
    with pytest.raises(ServingError):
        backend.complete(MESSAGES, SAMPLING)


# ---------------------------------------------------------------------------
# Token counts


def test_counts_are_whitespace_word_counts(tmp_path: Path) -> None:
    path = write_recording(tmp_path / "counts.json", [entry(FIRST_TEXT), entry("")])
    backend = replay_cls()("fixture/replay-a", recording=path)
    # MESSAGES, counted by hand: 3 + 6 words. FIRST_TEXT: ```, int, fixture_first(), {, return, 1;, }, ```.
    assert backend.complete(MESSAGES, SAMPLING) == Completion(text=FIRST_TEXT, prompt_tokens=9, completion_tokens=8)
    assert backend.complete([], SAMPLING) == Completion(text="", prompt_tokens=0, completion_tokens=0)


def test_docs_label_recordings_synthetic_and_counts_approximate() -> None:
    module = replay_module()
    cls = module.ReplayBackend
    for obj in (module, cls):
        assert "synthetic" in (inspect.getdoc(obj) or "").lower(), obj
    docs = " ".join(inspect.getdoc(obj) or "" for obj in (module, cls, cls.complete))
    words = " ".join(docs.lower().split())
    assert "approximat" in words, docs
    assert "never measurement" in words, docs


# ---------------------------------------------------------------------------
# Refused recordings: ValueError naming the file, when the backend is built


def raw_json(data: Any) -> bytes:
    """Return `data` as ASCII JSON bytes."""
    return json.dumps(data).encode("ascii")


def labeled(completions: Any) -> bytes:
    """Return a recording labeled synthetic whose completions value is `completions`."""
    return raw_json({"synthetic": True, "completions": completions})


BAD_RECORDINGS = {
    "not-json": b"synthetic: true\ncompletions: []\n",
    "non-ascii-bytes": b'{"synthetic": true, "completions": [{"text": "caf\xc3\xa9"}]}\n',
    "bare-list": raw_json([{"text": "Fixture text."}]),
    "label-missing": raw_json({"completions": [{"text": "Fixture text."}]}),
    "label-false": raw_json({"synthetic": False, "completions": [{"text": "Fixture text."}]}),
    "label-not-boolean": raw_json({"synthetic": "yes", "completions": [{"text": "Fixture text."}]}),
    "completions-missing": raw_json({"synthetic": True}),
    "completions-not-a-list": labeled({"text": "Fixture text."}),
    "entry-not-an-object": labeled(["Fixture text."]),
    "text-missing": labeled([{"messages": []}]),
    "text-not-a-string": labeled([{"text": 7}]),
    "messages-not-a-list": labeled([{"text": "Fixture text.", "messages": "Fixture user text."}]),
    "message-without-content": labeled([{"text": "Fixture text.", "messages": [{"role": "user"}]}]),
    "message-role-not-a-string": labeled([{"text": "Fixture text.", "messages": [{"role": 1, "content": "x"}]}]),
    # A misspelled key would silently turn the message check off, so unknown entry keys are refused.
    "unknown-entry-key": labeled([{"text": "Fixture text.", "mesages": []}]),
}


@pytest.mark.parametrize("raw", BAD_RECORDINGS.values(), ids=BAD_RECORDINGS.keys())
def test_a_bad_recording_is_refused_when_the_backend_is_built(tmp_path: Path, raw: bytes) -> None:
    path = tmp_path / "bad-recording.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError) as info:
        replay_cls()("fixture/replay-a", recording=path)
    assert "bad-recording.json" in str(info.value)


def test_a_missing_recording_file_fails_when_the_backend_is_built(tmp_path: Path) -> None:
    with pytest.raises((OSError, ValueError)) as info:
        replay_cls()("fixture/replay-a", recording=tmp_path / "no-such-recording.json")
    assert "no-such-recording.json" in str(info.value)
