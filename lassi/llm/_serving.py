"""The serving record: what a served backend's run records about its server (task P17.3; bible Serving Rules).

serving_record builds the record that provenance.json holds as `serving`:
the backend's base_url, the server's model entry as sent (every field kept but
the ones new on every GET, VOLATILE_FIELDS), the version the server reports,
and the request that version came from. The version is read only from a route
the entry's owned_by names (VERSION_ENDPOINTS), at the server root (base_url
without a final /v1 segment); version_from is null when no request was sent
and set when one was, so a null version beside it means asked, not reported.
A record that would hold the API key, as sent or escaped, is refused (Agent
Rule 12). serving_line gives run.md's one-line Server summary.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit

from lassi.llm._http import ServingError, redact

# Entry fields that change on every GET (plans/spikes/p17-frameworks.md, Results 4): `created` in every server's
# entry, and vLLM's `permission` list, each item with a fresh id and time beside the server's access defaults.
VOLATILE_FIELDS = frozenset({"created", "permission"})

# owned_by -> (path at the server root, field of its JSON reply that holds the version). Each server sets owned_by
# to its own name; the routes and fields are source-read in plans/spikes/p17-frameworks.md, Results 4.
VERSION_ENDPOINTS: dict[str, tuple[str, str]] = {
    "vllm": ("/version", "version"),
    "sglang": ("/server_info", "version"),
    "llamacpp": ("/props", "build_info"),
}

# The entry's context-length fields that run.md shows, in this order, when the entry has them.
CONTEXT_FIELDS = ("max_model_len", "max_context_len", "max_prompt_len")

_V1 = "/v1"


def server_root(base_url: str) -> str | None:
    """Return base_url without its final /v1 path segment, or None when its path does not end in one.

    For example, http://h:8000/v1 gives http://h:8000 and http://h:8000/prefix/v1
    gives http://h:8000/prefix; http://h:8000, .../v10, .../apiv1, and
    .../v1/extra give None, and then no version request is sent.
    """
    if not urlsplit(base_url).path.endswith(_V1):
        return None
    return base_url[: -len(_V1)]


def stable_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deep copy of the model entry without VOLATILE_FIELDS; every other field is kept as sent."""
    return {key: copy.deepcopy(value) for key, value in entry.items() if key not in VOLATILE_FIELDS}


def version_endpoint(entry: Mapping[str, Any]) -> tuple[str, str] | None:
    """Return (path, field) of the version route the entry's owned_by names, or None for any other owned_by."""
    owner = entry.get("owned_by")
    return VERSION_ENDPOINTS.get(owner) if isinstance(owner, str) else None


def serving_record(
    base_url: str,
    entry: Mapping[str, Any],
    asked: tuple[str, str] | None,
    version: str | None,
    secrets: Sequence[str],
) -> dict[str, Any]:
    """Return {base_url, model, version, version_from}; raise ServingError when it would hold a secret.

    `asked` is the (path, field) of the version request that was sent, or
    None when none was; version_from is then "GET <path> <field>" or null.
    The record is rendered as JSON once, with ASCII escapes as the run files
    write it, and refused when any of `secrets` (the API key) appears in it,
    as sent or escaped (lassi.llm._http.redact). The message quotes nothing.
    """
    record = {
        "base_url": base_url,
        "model": stable_entry(entry),
        "version": version,
        "version_from": None if asked is None else f"GET {asked[0]} {asked[1]}",
    }
    text = json.dumps(record, ensure_ascii=True)
    if redact(text, secrets) != text:
        raise ServingError(
            "the serving record would hold the API key, which the server sent back in its model entry or version; "
            "the run stops so that no key is written"
        )
    return record


def serving_line(record: Mapping[str, Any]) -> str:
    """Return run.md's Server text: labeled pairs joined by "; ".

    owned_by and version come first, then each of CONTEXT_FIELDS the entry
    has, then meta.n_ctx when the entry's meta holds it. A missing or null
    value shows as "-", any other as str(value); for example "owned_by vllm;
    version 0.31.0; max_model_len 32768". The labels keep an owned_by that
    is a model namespace (Ollama's) from reading as a server name.
    """
    model = record["model"]
    pairs = [("owned_by", model.get("owned_by")), ("version", record.get("version"))]
    pairs += [(name, model[name]) for name in CONTEXT_FIELDS if name in model]
    meta = model.get("meta")
    if isinstance(meta, Mapping) and "n_ctx" in meta:
        pairs.append(("meta.n_ctx", meta["n_ctx"]))
    return "; ".join(f"{label} {'-' if value is None else value}" for label, value in pairs)
