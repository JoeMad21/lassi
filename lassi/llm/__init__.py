"""LLM backends: the LLMBackend components (bible Component Interfaces, Model Serving).

Importing this package registers four backends in
lassi.core.registry.DEFAULT_REGISTRY under the interface "LLMBackend":

- "mock" (MockBackend): replies with the bench item's reference target in
  FILE blocks, for runs without a model host.
- "openai_compat" (OpenAICompatBackend): any OpenAI-compatible chat server.
- "ollama" (OllamaBackend): an Ollama server, with unload().
- "replay" (ReplayBackend): replies, in order, with the scripted synthetic
  completions of a recording file, for replay tests without a model host.

Every backend has the class attributes `name` and `capabilities`, an instance
attribute `model_id`, and `complete(messages, sampling) -> Completion`. Its
repr shows the class, model_id, and base_url only. Any failure talking to a
model server raises ServingError.

The construction convention, which lassi.core.registry states too: The
runner, never the registry, constructs components. A component bound by a
kind section is built as `factory(**binding.config)`; an LLM backend as
`factory(model_id)`, with keyword settings left at their defaults unless the
caller passes them. The keyword settings of the HTTP backends are base_url,
timeout_s, and, for openai_compat, api_key_env; the replay backend's is
recording, a file it reads when built. Construction sends no request.

model_info(backend, sampling) returns the Trial `model` field (bible Result
Record); it is how the sampling parameters land in the record.
"""

from __future__ import annotations

from typing import Protocol

from lassi.core.interfaces import LLMBackend, Sampling
from lassi.core.record import ModelInfo
from lassi.llm import mock, ollama, openai_compat, replay
from lassi.llm._http import ServingError
from lassi.llm.mock import MockBackend
from lassi.llm.ollama import OllamaBackend
from lassi.llm.openai_compat import OpenAICompatBackend
from lassi.llm.replay import ReplayBackend

__all__ = [
    "MockBackend",
    "OllamaBackend",
    "OpenAICompatBackend",
    "ReplayBackend",
    "ServedBackend",
    "ServingError",
    "mock",
    "model_info",
    "ollama",
    "openai_compat",
    "replay",
]


class ServedBackend(LLMBackend, Protocol):
    """An LLMBackend with the model_id attribute every backend in this package has."""

    model_id: str


def model_info(backend: ServedBackend, sampling: Sampling) -> ModelInfo:
    """Return the Trial `model` field: the backend's registry name, its model_id, and `sampling`."""
    return ModelInfo(backend=backend.name, id=backend.model_id, sampling=sampling)
