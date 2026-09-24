"""The HTTP transport and reply checks shared by the model-server backends, and ServingError.

request_json sends one JSON request with urllib and a timeout and returns the
decoded JSON reply (a JSONReply, which also keeps the status and the body text
for error messages). A loopback host (127.0.0.1, localhost, ::1) is reached
through an opener with no proxies, so a system proxy never catches local
traffic; any other host goes through urllib's proxy handling, which honors the
proxy environment as it is when the request is made. Redirects are never
followed: a 3xx reply is an error that asks for the server's final URL, so a
chat request is never turned into a GET or answered by a server that check()
did not see.

Each request is logged to the "lassi.llm" logger at DEBUG with the method,
the URL path, and the status only; headers and bodies are never logged.
Every urllib, socket, timeout, or JSON error becomes ServingError, raised with
no chained exception. Its message is plain ASCII, names the method, the URL
path, and the status when there is one, and quotes at most the first 500
characters of any server text it shows (a reply body or a transport error);
a list of served model ids is cut the same way, and the ids left out are
counted. Before any server text reaches a message, each extra header value and
each secret the caller names (such as an API key) is replaced by <redacted>,
whether the server echoes it as sent or with characters escaped the way JSON,
HTML, or URL encoding writes them (see redact).
"""

from __future__ import annotations

import http.client
import json
import logging
import math
import re
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

LOG = logging.getLogger("lassi.llm")
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
QUOTE_LIMIT = 500
REDACTED = "<redacted>"
# A served model entry is copied for every caller of check(); a deeper one is refused, so copying never recurses far.
MAX_ENTRY_DEPTH = 32
_SCHEMES = ("http", "https")
_REDIRECT_HINT = " (a redirect, which is never followed; set base_url to the server's final URL)"
# The two-character JSON escapes (PHP, for one, writes "/" as "\/") and the named HTML references.
_JSON_ESCAPES = {'"': '\\"', "\\": "\\\\", "/": "\\/", "\b": "\\b", "\f": "\\f", "\n": "\\n", "\r": "\\r", "\t": "\\t"}
_HTML_NAMES = {"&": "amp", "<": "lt", ">": "gt", '"': "quot", "'": "apos"}


class ServingError(RuntimeError):
    """Any failure talking to a model server; the message never holds an API key or a request header."""


def checked_base_url(url: str) -> str:
    """Return a backend's base URL without any trailing '/'; raise ValueError unless it is a plain http(s) URL.

    The URL needs a host and must have no user name, password, query, or
    fragment, because request paths are appended to it and credentials come
    only from environment variables (Agent Rule 12). The message never quotes
    the URL.
    """
    parts = urlsplit(url)
    if parts.scheme not in _SCHEMES or not parts.hostname:
        raise ValueError("base_url must be an http or https URL with a host")
    if "@" in parts.netloc:
        raise ValueError("base_url must not hold a user name or password; credentials come only from the environment")
    if "?" in url or "#" in url:
        raise ValueError("base_url must not have a query or a fragment; request paths are appended to it")
    try:
        port_ok = parts.port is None or 0 <= parts.port <= 65535
    except ValueError:  # urllib quotes the port text in this error, so it is never shown
        port_ok = False
    if not port_ok:
        raise ValueError("base_url must have a numeric port from 0 to 65535 when it names a port")
    return url.rstrip("/")


def checked_timeout(timeout_s: float) -> float:
    """Return timeout_s as a float; raise ValueError unless it is a finite number of seconds above zero."""
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or not 0 < timeout_s < math.inf:
        raise ValueError("timeout_s must be a finite number of seconds above zero")
    return float(timeout_s)


def redact(text: str, secrets: Sequence[str]) -> str:
    """Return `text` with each secret replaced by <redacted>, as sent or with any characters escaped.

    A server may echo a key back escaped, so after the verbatim secret is
    replaced, each character of it also matches its escaped forms (see
    _char_forms), in any mix: PHP-style JSON ("\\/"), Go's HTML-safe JSON
    ("\\u003c"), an HTML page ("&amp;"), or a percent-encoded URL ("%2B").
    """
    for secret in secrets:
        if secret:
            text = text.replace(secret, REDACTED)
            text = re.sub("".join(_char_forms(char) for char in secret), REDACTED, text)
    return text


def _char_forms(char: str) -> str:
    """Return a regex for one character as sent or as JSON, HTML, or URL encoding writes it, hex in either case.

    The escaped forms: the JSON escape \\uXXXX (a surrogate pair above
    U+FFFF) and the two-character ones such as \\/; a decimal, hex, or named
    HTML reference; %XX for each UTF-8 byte, and + for a space. A character
    that starts some of its own escapes (a backslash, '%', '&') matches raw
    only where none of them does, so each character matches in at most one way
    at any position and a failed match never retries another split of the text
    (redact replaces the verbatim secret first).

    For example, "/" matches /, \\/, \\u002f, %2F, &#47;, and &#x2f;.
    """
    utf16 = char.encode("utf-16-be", "surrogatepass")
    escapes = [
        "".join(rf"\\u(?i:{utf16[i:i + 2].hex()})" for i in range(0, len(utf16), 2)),
        "".join(f"%(?i:{byte:02x})" for byte in char.encode("utf-8", "surrogatepass")),
        f"&#0*{ord(char)};",
        f"&#(?i:x0*{ord(char):x});",
    ]
    if char in _JSON_ESCAPES:
        escapes.append(re.escape(_JSON_ESCAPES[char]))
    if char in _HTML_NAMES:
        escapes.append(f"&(?i:{_HTML_NAMES[char]});")
    if char == " ":
        escapes.append(r"\+")
    raw = re.escape(char)
    if char in "\\%&":
        # The raw character must not start one of its own escapes, or a match could take two paths.
        raw = f"(?!{'|'.join(escapes)}){raw}"
    return f"(?:{raw}|{'|'.join(escapes)})"


def quote(text: bytes | str, secrets: Sequence[str] = ()) -> str:
    """Return server text for an error message: secrets redacted, then at most 500 characters, as an ASCII literal."""
    text = redact(text.decode("utf-8", "replace") if isinstance(text, bytes) else text, secrets)
    shown = ascii(text[:QUOTE_LIMIT])
    return shown if len(text) <= QUOTE_LIMIT else f"{shown} (first {QUOTE_LIMIT} of {len(text)} characters)"


@dataclass(frozen=True)
class JSONReply:
    """A 2xx reply: `where` is 'METHOD /path', `data` the decoded JSON, and `text` the body as received.

    `secrets` are the values redacted from any error that quotes the reply.
    """

    where: str
    status: int
    data: Any
    text: str = field(repr=False)
    secrets: tuple[str, ...] = field(default=(), repr=False)

    def error(self, problem: str) -> ServingError:
        """Return the ServingError for a reply that lacks something the backend needs, quoting the body."""
        return ServingError(
            f"{self.where} returned HTTP {self.status}, but {problem}; body: {quote(self.text, self.secrets)}"
        )


def request_json(
    method: str,
    url: str,
    *,
    timeout_s: float,
    body: Any = None,
    headers: Mapping[str, str] | None = None,
    secrets: Sequence[str] = (),
) -> JSONReply:
    """Send one request and return the decoded 2xx JSON reply; raise ServingError on any failure.

    `body`, when given, is sent as JSON. `headers` are sent besides
    Content-Type: application/json and are never logged or sent on to a
    redirect target. Their values and `secrets` (such as the API key inside an
    Authorization header) are redacted from every server text an error quotes.
    """
    where = f"{method} {urlsplit(url).path or '/'}"
    hidden = (*(headers or {}).values(), *secrets)
    request = _build_request(where, method, url, body, headers)
    status, raw = _send(request, url, where, timeout_s, hidden)
    LOG.debug("%s -> %s", where, status)
    try:
        text: str | None = raw.decode("utf-8-sig")  # a leading byte order mark is dropped, as JSON readers allow
        data = json.loads(text)
    except (ValueError, RecursionError):
        text = None  # raised below, outside the handler, so the error chains no decoder error holding the body
    if text is None:
        raise ServingError(f"{where} returned HTTP {status} with a body that is not JSON; body: {quote(raw, hidden)}")
    return JSONReply(where=where, status=status, data=data, text=text, secrets=hidden)


def _send(
    request: urllib.request.Request, url: str, where: str, timeout_s: float, hidden: Sequence[str]
) -> tuple[int, bytes]:
    """Return the status and body of a 2xx reply; raise ServingError, with no chained exception, otherwise."""
    try:
        with _opener_for(url).open(request, timeout=timeout_s) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        LOG.debug("%s -> %s", where, exc.code)
        hint = _REDIRECT_HINT if 300 <= exc.code < 400 else ""
        problem = f"returned HTTP {exc.code}{hint}; body: {quote(_error_body(exc), hidden)}"
    except (OSError, http.client.HTTPException) as exc:
        LOG.debug("%s -> no reply", where)
        problem = f"got no reply: {type(exc).__name__}: {quote(str(exc), hidden)}"
    except ValueError as exc:
        # http.client quotes a refused header value in this error, so its text is never shown.
        LOG.debug("%s -> not sent", where)
        problem = f"could not be sent ({type(exc).__name__})"
    # Raised here, outside the handlers, so no urllib error holding server or header text is chained to it.
    raise ServingError(f"{where} {problem}")


def _build_request(
    where: str, method: str, url: str, body: Any, headers: Mapping[str, str] | None
) -> urllib.request.Request:
    """Return the urllib request; the extra headers are unredirected so no redirect carries them to another host."""
    scheme = urlsplit(url).scheme
    if scheme not in _SCHEMES:
        raise ServingError(f"{where}: the URL scheme must be http or https, not {scheme!a}")
    try:
        data = None if body is None else json.dumps(body, allow_nan=False).encode("utf-8")
        request: urllib.request.Request | None = urllib.request.Request(url, data=data, method=method)
    except (TypeError, ValueError) as exc:
        problem, request = f"{type(exc).__name__}: {exc}", None
    # Raised here, outside the handler, so the error chains no exception holding request text.
    if request is None:
        raise ServingError(f"{where}: the request could not be built: {quote(problem)}")
    request.add_header("Content-Type", "application/json")
    for name, value in (headers or {}).items():
        request.add_unredirected_header(name, value)
    return request


class _NoRedirects(urllib.request.HTTPRedirectHandler):
    """A redirect handler that follows nothing, so a 3xx reply reaches request_json as an HTTPError."""

    def redirect_request(
        self, req: urllib.request.Request, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> None:
        """Return None, which makes urllib raise HTTPError for the 3xx reply instead of following it."""
        return None


def _opener_for(url: str) -> urllib.request.OpenerDirector:
    """Return an opener that follows no redirect: no proxies for a loopback host, the proxy environment otherwise."""
    if urlsplit(url).hostname in LOOPBACK_HOSTS:
        return urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirects)
    return urllib.request.build_opener(_NoRedirects)


def _error_body(error: urllib.error.HTTPError) -> bytes:
    """Return the body of a non-2xx reply, or b'' when there is none or it cannot be read."""
    if error.fp is None:
        return b""
    try:
        return error.read()
    except (OSError, http.client.HTTPException):
        return b""
    finally:
        error.close()


def pick(data: Any, *keys: str | int) -> Any:
    """Return the value at `keys` (object keys, array indexes) inside decoded JSON, or None when a step is missing."""
    for key in keys:
        if isinstance(key, int) and isinstance(data, list) and 0 <= key < len(data):
            data = data[key]
        elif isinstance(key, str) and isinstance(data, dict) and key in data:
            data = data[key]
        else:
            return None
    return data


def reply_text(reply: JSONReply, *keys: str | int) -> str:
    """Return the reply text at `keys`; raise ServingError when it is missing, null, or not a string."""
    text = pick(reply.data, *keys)
    if not isinstance(text, str):
        raise reply.error(f"it has no text at {_dotted(keys)}")
    return text


def token_count(reply: JSONReply, *keys: str) -> int:
    """Return the token count at `keys`; raise ServingError unless the server sent an integer >= 0 (never guessed)."""
    value = pick(reply.data, *keys)
    if type(value) is not int or value < 0:
        raise reply.error(f"it has no token count at {_dotted(keys)}")
    return value


def served_entry(reply: JSONReply, model_id: str) -> dict[str, Any]:
    """Return the entry of an OpenAI-style model list whose id is exactly `model_id`.

    The reply must be `{"data": [{"id": ...}, ...]}`, and the entry may nest
    arrays and objects at most MAX_ENTRY_DEPTH levels deep (the entry itself is
    the first). When no entry has the id, the ServingError names the expected
    id and the served ids, sorted and cut as _id_list says.
    """
    entries = pick(reply.data, "data")
    if not isinstance(entries, list) or not all(isinstance(pick(entry, "id"), str) for entry in entries):
        raise reply.error('it is not a model list {"data": [{"id": ...}, ...]}')
    for entry in entries:
        if entry["id"] != model_id:
            continue
        if _nesting(entry) > MAX_ENTRY_DEPTH:
            raise reply.error(f"its entry for model {model_id!a} nests more than {MAX_ENTRY_DEPTH} levels deep")
        return entry
    ids = sorted(entry["id"] for entry in entries)
    raise ServingError(f"{reply.where} does not list model {model_id!a}; served ids: {_id_list(ids, reply.secrets)}")


def _id_list(ids: Sequence[str], secrets: Sequence[str]) -> str:
    """Return served ids for a refusal: each quoted (redacted and cut), ending when the list passes QUOTE_LIMIT.

    The first id is always shown; each later one only while the list stays
    within QUOTE_LIMIT characters. The ids left out are counted at the end.
    """
    shown: list[str] = []
    length = 0
    for served_id in ids:
        item = quote(served_id, secrets)
        length += len(item) + (2 if shown else 0)
        if shown and length > QUOTE_LIMIT:
            break
        shown.append(item)
    left_out = len(ids) - len(shown)
    listed = ", ".join(shown) or "none"
    return f"{listed} (and {left_out} more)" if left_out else listed


def _nesting(value: Any) -> int:
    """Return how many levels of arrays and objects nest in decoded JSON (0 for a scalar), without recursion."""
    depth, level = 0, [value]
    while True:
        containers = [item for item in level if isinstance(item, (list, dict))]
        if not containers:
            return depth
        depth += 1
        level = [child for item in containers for child in (item.values() if isinstance(item, dict) else item)]


def _dotted(keys: tuple[str | int, ...]) -> str:
    """Return a JSON location such as choices[0].message.content."""
    text = ""
    for key in keys:
        text += f"[{key}]" if isinstance(key, int) else f".{key}"
    return text.lstrip(".")
