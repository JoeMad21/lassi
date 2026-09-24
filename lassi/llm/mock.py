"""The mock LLM backend, registered as LLMBackend "mock" (bible Risks And Questions: the mock LLM mitigation).

It answers every request with the bench item's reference target files, so
the pipeline runs end to end without a model host. It declares the
`needs_reference` capability: before each trial the runner hands it the bench
item's reference target through with_reference.

Its reply takes one of two forms, and the runner picks the one the recipe's
extraction reads:

- FILE blocks (bible Harness Contract), exactly as render_file_blocks writes
  them: the default, read with the fence_tag fix on;
- one untagged fence holding the target's one file (with_untagged_fence): the
  form upstream's system prompts ask for, read by faithful extraction
  (fixes.fence_tag off). FILE blocks tag a .cu file `cuda`, which upstream's
  tag stripping would turn into `uda`, so a faithful run would not get the
  reference back from them.

Its token counts are approximations, never measurements: whitespace word
counts (len(text.split())) of the messages and of the reply.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from lassi.core.files import render_file_blocks
from lassi.core.interfaces import Completion, Message, Sampling
from lassi.core.registry import register
from lassi.llm._http import ServingError

# A run of three backticks: upstream's extraction reads the text between the first two.
FENCE = "```"


def untagged_fence(files: Mapping[str, str]) -> str:
    """Return one untagged fenced block holding the text of the one file in `files`.

    The block opens with a bare fence line and closes on its own line, so
    upstream's extraction (the text between the first two runs of three
    backticks, then its tag stripping) returns the file's text with one
    leading line feed. Raises ValueError when `files` does not hold exactly
    one file, or when its text holds a run of three backticks, which would
    end the block early.
    """
    if len(files) != 1:
        raise ValueError(f"one untagged fence holds one file, but the reference has {len(files)}")
    ((path, text),) = files.items()
    if FENCE in text:
        raise ValueError(f"{path} holds a run of three backticks, which would end an untagged fence early")
    end = "" if text.endswith("\n") else "\n"
    return f"{FENCE}\n{text}{end}{FENCE}\n"


@register("LLMBackend", "mock")
class MockBackend:
    """Replies with the reference target files; token counts are approximate word counts."""

    name = "mock"
    capabilities = frozenset({"chat", "needs_reference"})

    def __init__(self, model_id: str = "mock-reference", *, reference: Mapping[str, str] | None = None) -> None:
        """Keep `model_id`, a copy of `reference` (relative path -> text), and its FILE-block rendering.

        The reference is rendered here, so a bad one fails when the backend is
        built rather than in complete(): an empty mapping or a bad path raises
        ValueError from render_file_blocks. With no reference (None), complete()
        raises ServingError.
        """
        self.model_id = model_id
        self._files = None if reference is None else dict(reference)
        self._reply = None if reference is None else render_file_blocks(dict(reference))

    def __repr__(self) -> str:
        """Show the class and model_id only."""
        return f"{type(self).__name__}(model_id={self.model_id!r})"

    def with_reference(self, files: Mapping[str, str]) -> MockBackend:
        """Return a new backend with the same model_id whose reply is `files`, the bench item's reference target.

        The reply is in FILE blocks. Raises ValueError, as the constructor
        does, when `files` is empty or has a bad path.
        """
        return type(self)(self.model_id, reference=files)

    def with_untagged_fence(self) -> MockBackend:
        """Return a new backend with the same model_id and reference whose reply is one untagged fence.

        The runner asks for it when fixes.fence_tag is off (faithful
        extraction). Raises ServingError when the backend has no reference,
        and ValueError (untagged_fence) when the reference is not one file or
        holds a run of three backticks.
        """
        if self._files is None:
            raise ServingError(f"{self!r} has no reference files; call with_reference(files) first")
        backend = type(self)(self.model_id, reference=self._files)
        backend._reply = untagged_fence(self._files)
        return backend

    def complete(self, messages: Sequence[Message], sampling: Sampling) -> Completion:
        """Return the reference target in the reply's form; `messages` and `sampling` never change the text.

        The token counts are deterministic approximations, never measurements:
        prompt_tokens is the sum of len(content.split()) over the messages and
        completion_tokens is len(text.split()) of the reply. Raises
        ServingError when the backend has no reference.
        """
        if self._reply is None:
            raise ServingError(
                f"{self!r} has no reference files; call with_reference(files) with the bench item's "
                "reference target first"
            )
        prompt_tokens = sum(len(message.content.split()) for message in messages)
        return Completion(text=self._reply, prompt_tokens=prompt_tokens, completion_tokens=len(self._reply.split()))
