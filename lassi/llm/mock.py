"""The mock LLM backend, registered as LLMBackend "mock" (bible Risks And Questions: the mock LLM mitigation).

It answers every request with the bench item's reference target files in
FILE blocks (bible Harness Contract), exactly as render_file_blocks writes
them, so the pipeline runs end to end without a model host. It declares the
`needs_reference` capability: before each trial the runner hands it the bench
item's reference target through with_reference.

Its token counts are approximations, never measurements: whitespace word
counts (len(text.split())) of the messages and of the reply.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from lassi.core.files import render_file_blocks
from lassi.core.interfaces import Completion, Message, Sampling
from lassi.core.registry import register
from lassi.llm._http import ServingError


@register("LLMBackend", "mock")
class MockBackend:
    """Replies with the reference target files in FILE blocks; token counts are approximate word counts."""

    name = "mock"
    capabilities = frozenset({"chat", "needs_reference"})

    def __init__(self, model_id: str = "mock-reference", *, reference: Mapping[str, str] | None = None) -> None:
        """Keep `model_id` and a rendered copy of `reference` (relative path -> text).

        The reference is rendered here, so a bad one fails when the backend is
        built rather than in complete(): an empty mapping or a bad path raises
        ValueError from render_file_blocks. With no reference (None), complete()
        raises ServingError.
        """
        self.model_id = model_id
        self._reply = None if reference is None else render_file_blocks(dict(reference))

    def __repr__(self) -> str:
        """Show the class and model_id only."""
        return f"{type(self).__name__}(model_id={self.model_id!r})"

    def with_reference(self, files: Mapping[str, str]) -> MockBackend:
        """Return a new backend with the same model_id whose reply is `files`, the bench item's reference target.

        Raises ValueError, as the constructor does, when `files` is empty or has a bad path.
        """
        return type(self)(self.model_id, reference=files)

    def complete(self, messages: Sequence[Message], sampling: Sampling) -> Completion:
        """Return the reference target in FILE blocks; `messages` and `sampling` never change the text.

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
