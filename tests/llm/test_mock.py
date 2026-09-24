"""Tests for the mock LLM backend in lassi/llm/mock.py (P0.4).

The mock is the bible's mitigation for missing model hosts (Risks And
Questions) and the model arm of the P0 gate: it returns a bench item's
reference target wrapped in `// FILE:` blocks (Execution Backends, Harness
Contract), as render_file_blocks writes them. Its token counts are
deterministic whitespace word counts, documented as approximations. The
backend is registered as LLMBackend "mock" and declares `needs_reference`, so
the runner hands it each item's reference through with_reference. Every
expected text and count below was written by hand from those rules; no value
in this module is a measurement.
"""

from __future__ import annotations

import inspect

import pytest

from lassi import llm
from lassi.core import files as file_blocks
from lassi.core.interfaces import Completion, Message, Sampling
from lassi.core.registry import DEFAULT_REGISTRY

MockBackend = llm.mock.MockBackend
ServingError = llm.ServingError
render_file_blocks = file_blocks.render_file_blocks

SAMPLING = Sampling(temperature=0.2, top_p=0.9, max_tokens=1024)
OTHER_SAMPLING = Sampling(temperature=1.3, top_p=0.1, max_tokens=7)

REFERENCE = {
    "src/main.cu": '#include "kernels/scale.cuh"\nint main() { return 0; }\n',
    "kernels/scale.cuh": "__device__ float scale(float x);\n",
}
REFERENCE_TEXT = (
    "```cuda\n"
    "// FILE: kernels/scale.cuh\n"
    "__device__ float scale(float x);\n"
    "```\n"
    "\n"
    "```cuda\n"
    "// FILE: src/main.cu\n"
    '#include "kernels/scale.cuh"\n'
    "int main() { return 0; }\n"
    "```\n"
)
# Whitespace words of REFERENCE_TEXT, counted by hand: 9 in the first block and 13 in the second.
REFERENCE_WORDS = 22

MESSAGES = (
    Message(role="system", content="You translate code."),
    Message(role="user", content="Translate  main.cpp\nto CUDA.\n"),
)
# Whitespace words of MESSAGES, counted by hand: 3 + 4.
MESSAGE_WORDS = 7

OTHER_REFERENCE = {"main.cpp": "int main() {\n  return 1;\n}"}
OTHER_REFERENCE_TEXT = "```cpp\n// FILE: main.cpp\nint main() {\n  return 1;\n}\n```\n"

# A reference file holding a triple-backtick run: its block gets a four-backtick fence.
FENCED_REFERENCE = {"kernels/scale.cuh": "// Usage:\n// ```\n// out[i] = scale(in[i]);\n// ```\n"}
FENCED_REFERENCE_TEXT = (
    "````cuda\n// FILE: kernels/scale.cuh\n// Usage:\n// ```\n// out[i] = scale(in[i]);\n// ```\n````\n"
)
# Whitespace words of FENCED_REFERENCE_TEXT, counted by hand: ````cuda // FILE: kernels/scale.cuh (4),
# // Usage: (2), // ``` (2), // out[i] = scale(in[i]); (4), // ``` (2), ```` (1).
FENCED_REFERENCE_WORDS = 15


def test_hand_counts_match_the_word_rule() -> None:
    # Guards the hand-written constants: the rule is len(text.split()) per text.
    assert len(REFERENCE_TEXT.split()) == REFERENCE_WORDS
    assert sum(len(message.content.split()) for message in MESSAGES) == MESSAGE_WORDS
    assert render_file_blocks(REFERENCE) == REFERENCE_TEXT
    assert render_file_blocks(OTHER_REFERENCE) == OTHER_REFERENCE_TEXT
    assert render_file_blocks(FENCED_REFERENCE) == FENCED_REFERENCE_TEXT
    assert len(FENCED_REFERENCE_TEXT.split()) == FENCED_REFERENCE_WORDS


# ---------------------------------------------------------------------------
# Registration and shape


def test_registered_as_llm_backend_mock() -> None:
    entry = DEFAULT_REGISTRY.get("LLMBackend", "mock")
    assert entry.factory is MockBackend
    assert entry.capabilities == frozenset({"chat", "needs_reference"})
    assert llm.MockBackend is MockBackend


def test_name_and_capabilities() -> None:
    assert MockBackend.name == "mock"
    assert isinstance(MockBackend.capabilities, frozenset)
    assert MockBackend.capabilities == frozenset({"chat", "needs_reference"})


def test_constructor_signature() -> None:
    params = [(p.name, p.kind, p.default) for p in inspect.signature(MockBackend).parameters.values()]
    assert params == [
        ("model_id", inspect.Parameter.POSITIONAL_OR_KEYWORD, "mock-reference"),
        ("reference", inspect.Parameter.KEYWORD_ONLY, None),
    ]


def test_default_model_id() -> None:
    assert MockBackend().model_id == "mock-reference"
    assert MockBackend("mock-fixture").model_id == "mock-fixture"


def test_repr_shows_class_and_model_id_only() -> None:
    text = repr(MockBackend("mock-fixture", reference=REFERENCE))
    assert "MockBackend" in text and "mock-fixture" in text, text
    assert "scale(float x)" not in text and "kernels/scale.cuh" not in text, text


# ---------------------------------------------------------------------------
# complete


def test_complete_returns_the_reference_in_file_blocks() -> None:
    completion = MockBackend(reference=REFERENCE).complete(MESSAGES, SAMPLING)
    assert isinstance(completion, Completion)
    assert completion == Completion(text=REFERENCE_TEXT, prompt_tokens=MESSAGE_WORDS, completion_tokens=REFERENCE_WORDS)


def test_a_reference_with_a_triple_backtick_run_gets_a_longer_fence() -> None:
    completion = MockBackend(reference=FENCED_REFERENCE).complete(MESSAGES, SAMPLING)
    assert completion == Completion(
        text=FENCED_REFERENCE_TEXT, prompt_tokens=MESSAGE_WORDS, completion_tokens=FENCED_REFERENCE_WORDS
    )


def test_counts_are_whitespace_word_counts() -> None:
    backend = MockBackend(reference=OTHER_REFERENCE)
    messages = [Message("user", "  one\ttwo\n\nthree  "), Message("assistant", ""), Message("user", "four")]
    completion = backend.complete(messages, SAMPLING)
    assert completion.prompt_tokens == 4
    # ```cpp, //, FILE:, main.cpp, int, main(), {, return, 1;, }, ```
    assert completion.completion_tokens == 11 == len(OTHER_REFERENCE_TEXT.split())
    assert backend.complete([], SAMPLING).prompt_tokens == 0


def test_complete_is_deterministic() -> None:
    backend = MockBackend(reference=REFERENCE)
    assert backend.complete(MESSAGES, SAMPLING) == backend.complete(MESSAGES, SAMPLING)
    assert MockBackend(reference=REFERENCE).complete(MESSAGES, SAMPLING) == backend.complete(MESSAGES, SAMPLING)


def test_messages_and_sampling_do_not_change_the_text() -> None:
    backend = MockBackend(reference=REFERENCE)
    first = backend.complete(MESSAGES, SAMPLING)
    other = backend.complete([Message("user", "Something else entirely.")], OTHER_SAMPLING)
    assert other.text == first.text == REFERENCE_TEXT
    assert other.completion_tokens == first.completion_tokens
    assert other.prompt_tokens == 3


def test_counts_are_documented_as_approximations() -> None:
    docs = " ".join(inspect.getdoc(obj) or "" for obj in (llm.mock, MockBackend, MockBackend.complete))
    words = " ".join(docs.lower().split())
    assert "approximat" in words, docs
    assert "never measurement" in words, docs


def test_no_reference_raises_serving_error_naming_with_reference() -> None:
    with pytest.raises(ServingError) as info:
        MockBackend().complete(MESSAGES, SAMPLING)
    assert "with_reference" in str(info.value)


def test_the_backend_keeps_its_own_copy_of_the_reference() -> None:
    files = dict(REFERENCE)
    backend = MockBackend(reference=files)
    files["src/main.cu"] = "changed\n"
    files["extra.cu"] = "added\n"
    assert backend.complete(MESSAGES, SAMPLING).text == REFERENCE_TEXT


# ---------------------------------------------------------------------------
# with_reference


def test_with_reference_returns_a_new_backend() -> None:
    original = MockBackend("mock-fixture")
    fed = original.with_reference(REFERENCE)
    assert type(fed) is MockBackend
    assert fed is not original
    assert fed.model_id == "mock-fixture"
    assert fed.complete(MESSAGES, SAMPLING).text == REFERENCE_TEXT
    with pytest.raises(ServingError):
        original.complete(MESSAGES, SAMPLING)


def test_with_reference_replaces_and_leaves_the_original_alone() -> None:
    original = MockBackend(reference=REFERENCE)
    fed = original.with_reference(OTHER_REFERENCE)
    assert fed.complete(MESSAGES, SAMPLING).text == OTHER_REFERENCE_TEXT
    assert original.complete(MESSAGES, SAMPLING).text == REFERENCE_TEXT


def test_with_reference_keeps_its_own_copy() -> None:
    files = dict(OTHER_REFERENCE)
    fed = MockBackend().with_reference(files)
    files["main.cpp"] = "changed\n"
    assert fed.complete(MESSAGES, SAMPLING).text == OTHER_REFERENCE_TEXT


def test_with_reference_signature() -> None:
    assert list(inspect.signature(MockBackend.with_reference).parameters) == ["self", "files"]


@pytest.mark.parametrize("reference", [{"../outside.cu": "int x;\n"}, {}], ids=["bad-path", "empty"])
def test_a_bad_reference_is_refused_when_the_backend_is_built(reference: dict[str, str]) -> None:
    # The reference goes through render_file_blocks when the backend is built, so a bad one fails there,
    # before any complete() call: a path that escapes the build directory, or no file at all.
    with pytest.raises(ValueError):
        MockBackend(reference=reference)
    with pytest.raises(ValueError):
        MockBackend().with_reference(reference)
