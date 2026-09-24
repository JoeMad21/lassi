"""The OpenAI-compatible chat backend, registered as LLMBackend "openai_compat" (bible Model Serving).

It talks to any server with the OpenAI chat API, such as furiosa-llm or
vLLM. Before the first chat request, check() confirms that `<base_url>/models`
lists the exact model id (bible Serving Rules: port 8123, confirm /v1/models
returns the expected id before the first request). Messages and sampling are
sent unchanged, never truncated, and the token counts are the server's, never
estimated.

API keys (Agent Rule 12): api_key_env names an environment variable, and the
key is read from os.environ each time a request is made and sent as
`Authorization: Bearer <key>`. The key is never stored in an attribute and
never reaches a repr, a log, an error message, or a record: it is redacted
from any server text an error quotes, even when the server echoes it back.
While a request is being sent the key is a local variable of the sending
frames, so a traceback printer that dumps local variables (such as pytest
--showlocals) must stay off for runs that use a key.
"""

from __future__ import annotations

import copy
import os
import re
from collections.abc import Sequence
from typing import Any

from lassi.core.interfaces import Completion, Message, Sampling
from lassi.core.registry import register
from lassi.llm._http import (
    JSONReply,
    ServingError,
    checked_base_url,
    checked_timeout,
    reply_text,
    request_json,
    served_entry,
    token_count,
)

DEFAULT_BASE_URL = "http://127.0.0.1:8123/v1"

# An API key must be visible ASCII so it fits in a header; http.client would otherwise quote it in an error.
_KEY_TEXT = re.compile(r"[\x21-\x7e]+")
# api_key_env holds a variable name; this refuses a key value passed by mistake, such as "sk-...".
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@register("LLMBackend", "openai_compat")
class OpenAICompatBackend:
    """Chats with an OpenAI-compatible server once it has confirmed that the server lists model_id.

    Construction sends nothing. complete() calls check() first, so no chat
    request ever reaches a server that lacks the model.
    """

    name = "openai_compat"
    capabilities = frozenset({"chat", "model_check"})

    def __init__(
        self,
        model_id: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        api_key_env: str | None = None,
        timeout_s: float = 600.0,
    ) -> None:
        """Keep the settings; base_url loses any trailing '/'. No key is read until a request is made.

        Raises ValueError for a base_url that is not a plain http(s) URL (see
        lassi.llm._http.checked_base_url), such as one with a password or a
        query; for a timeout_s that is not a finite number above zero; and for
        an api_key_env that is not an environment variable name, such as a key
        passed by mistake (the message never shows the value).
        """
        if api_key_env is not None and not (isinstance(api_key_env, str) and _ENV_NAME.fullmatch(api_key_env)):
            raise ValueError(
                "api_key_env must name an environment variable (letters, digits, and '_', not starting with a "
                "digit); the value is not shown, since it may be a key"
            )
        self.model_id = model_id
        self.base_url = checked_base_url(base_url)
        self.api_key_env = api_key_env
        self.timeout_s = checked_timeout(timeout_s)
        self._entry: dict[str, Any] | None = None

    def __repr__(self) -> str:
        """Show the class, model_id, and base_url only."""
        return f"{type(self).__name__}(model_id={self.model_id!r}, base_url={self.base_url!r})"

    def check(self) -> dict[str, Any]:
        """Return the entry of GET <base_url>/models whose id is exactly model_id; raise ServingError otherwise.

        The refusal names the expected id and the served ids, sorted; an entry
        nested too deeply to copy safely is refused too (_http.served_entry).
        The first success is cached and a failure is not. The entry keeps every
        field the server sent (vLLM-style servers add max_model_len), so the
        runner can record them. The cache takes no lock: threads that call check() before
        the first success may each send the GET, and all get the same entry.
        """
        if self._entry is None:
            self._entry = served_entry(self._request("GET", "/models"), self.model_id)
        return copy.deepcopy(self._entry)

    def complete(self, messages: Sequence[Message], sampling: Sampling) -> Completion:
        """Return the server's reply text and token counts for `messages` under `sampling`.

        Calls check() first. POSTs <base_url>/chat/completions with stream off.
        A missing or null content or token count, a non-2xx status, or a body
        that is not JSON raises ServingError.
        """
        self.check()
        body = {
            "model": self.model_id,
            "messages": [{"role": message.role, "content": message.content} for message in messages],
            "temperature": float(sampling.temperature),
            "top_p": float(sampling.top_p),
            "max_tokens": sampling.max_tokens,
            "stream": False,
        }
        reply = self._request("POST", "/chat/completions", body)
        return Completion(
            text=reply_text(reply, "choices", 0, "message", "content"),
            prompt_tokens=token_count(reply, "usage", "prompt_tokens"),
            completion_tokens=token_count(reply, "usage", "completion_tokens"),
        )

    def _request(self, method: str, path: str, body: Any = None) -> JSONReply:
        """Send one request to <base_url><path>, with the API key when api_key_env is set, redacted from errors."""
        key = self._key()
        headers = {} if key is None else {"Authorization": "Bearer " + key}
        secrets = () if key is None else (key,)
        url = self.base_url + path
        return request_json(method, url, body=body, headers=headers, secrets=secrets, timeout_s=self.timeout_s)

    def _key(self) -> str | None:
        """Return the API key read from the environment now, or None when api_key_env is not set."""
        if self.api_key_env is None:
            return None
        key = os.environ.get(self.api_key_env, "")
        if not key:
            raise ServingError(f"the API key variable {self.api_key_env!a} (api_key_env) is not set or is empty")
        if not _KEY_TEXT.fullmatch(key):
            raise ServingError(
                f"the API key variable {self.api_key_env!a} (api_key_env) holds a value that cannot be sent in a "
                "header; it must be visible ASCII with no spaces"
            )
        return key
