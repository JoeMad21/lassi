"""The Ollama chat backend, registered as LLMBackend "ollama" (bible Model Serving, the Ollama arm).

Before the first chat request, check() confirms that Ollama's
OpenAI-compatible model list at `<base_url>/v1/models` holds the exact model
id, full tag included, such as `name:tag` (bible Serving Rules). complete()
chats through `<base_url>/api/chat` with the sampling parameters in `options`
(max_tokens as num_predict) and returns Ollama's own token counts
(prompt_eval_count and eval_count); a missing count raises ServingError and is
never estimated. unload() asks Ollama to drop the model from memory (bible
LASSI quirk table: Ollama unload before each execution, for Ollama arms only).
The backend declares `unload_before_run` (lassi.core.capabilities), so the
runner asks it to unload at trial start and run_loop right before each run of
an attempt.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

from lassi.core.capabilities import UNLOAD_BEFORE_RUN
from lassi.core.interfaces import Completion, Message, Sampling
from lassi.core.registry import register
from lassi.llm._http import (
    checked_base_url,
    checked_timeout,
    reply_text,
    request_json,
    served_entry,
    token_count,
)

DEFAULT_BASE_URL = "http://127.0.0.1:11434"


@register("LLMBackend", "ollama")
class OllamaBackend:
    """Chats with an Ollama server once it has confirmed that the server lists model_id, and unloads it on request.

    Construction sends nothing. complete() calls check() first, so no chat
    request ever reaches a server that lacks the model.
    """

    name = "ollama"
    capabilities = frozenset({"chat", "model_check", "unload", UNLOAD_BEFORE_RUN})

    def __init__(self, model_id: str, *, base_url: str = DEFAULT_BASE_URL, timeout_s: float = 600.0) -> None:
        """Keep the settings; base_url loses any trailing '/'.

        Raises ValueError for a base_url that is not a plain http(s) URL (see
        lassi.llm._http.checked_base_url), such as one with a password or a
        query, and for a timeout_s that is not a finite number above zero.
        """
        self.model_id = model_id
        self.base_url = checked_base_url(base_url)
        self.timeout_s = checked_timeout(timeout_s)
        self._entry: dict[str, Any] | None = None

    def __repr__(self) -> str:
        """Show the class, model_id, and base_url only."""
        return f"{type(self).__name__}(model_id={self.model_id!r}, base_url={self.base_url!r})"

    def check(self) -> dict[str, Any]:
        """Return the entry of GET <base_url>/v1/models whose id is exactly model_id; raise ServingError otherwise.

        The refusal names the expected id and the served ids, sorted; an entry
        nested too deeply to copy safely is refused too (_http.served_entry).
        The first success is cached and a failure is not. The cache takes no lock: threads
        that call check() before the first success may each send the GET, and
        all get the same entry.
        """
        if self._entry is None:
            reply = request_json("GET", f"{self.base_url}/v1/models", timeout_s=self.timeout_s)
            self._entry = served_entry(reply, self.model_id)
        return copy.deepcopy(self._entry)

    def complete(self, messages: Sequence[Message], sampling: Sampling) -> Completion:
        """Return Ollama's reply text and token counts for `messages` under `sampling`.

        Calls check() first. POSTs <base_url>/api/chat with stream off. A
        missing or null content or count, a non-2xx status, or a body that is
        not JSON raises ServingError.
        """
        self.check()
        body = {
            "model": self.model_id,
            "messages": [{"role": message.role, "content": message.content} for message in messages],
            "stream": False,
            "options": {
                "temperature": float(sampling.temperature),
                "top_p": float(sampling.top_p),
                "num_predict": sampling.max_tokens,
            },
        }
        reply = request_json("POST", f"{self.base_url}/api/chat", body=body, timeout_s=self.timeout_s)
        return Completion(
            text=reply_text(reply, "message", "content"),
            prompt_tokens=token_count(reply, "prompt_eval_count"),
            completion_tokens=token_count(reply, "eval_count"),
        )

    def unload(self) -> None:
        """Ask Ollama to unload the model now: POST <base_url>/api/generate with keep_alive 0."""
        request_json(
            "POST",
            f"{self.base_url}/api/generate",
            body={"model": self.model_id, "keep_alive": 0},
            timeout_s=self.timeout_s,
        )
