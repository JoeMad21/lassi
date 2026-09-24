"""Shared fixtures for the LLM backend tests (P0.4).

`stub_server` runs a standard-library HTTP server on 127.0.0.1 with a port the
OS picks, in a daemon thread. Tests configure a canned reply per method and
path (served model ids, chat replies, status codes, extra reply headers such as
a redirect Location, malformed bodies, delays, and raw malformed responses),
and the stub records every request it receives: method, request target, path,
headers, raw body, and the body parsed as JSON. A reply can echo a request
header: every occurrence of a marker the test picks is replaced in its body or
raw response by the value of that header as received, the way some servers
quote a rejected API key.
Paths without a configured reply get a 404. The server shuts down after the
test.

`no_network` (autouse) refuses name lookups of any host other than the
loopback names, so a backend that tries to reach the network fails the test
instead of leaving the machine; every URL a test builds is the stub's
127.0.0.1 address (or, in the proxy tests, an unresolvable name routed through
the stub as a proxy). No value served by the stub is a measurement.
"""

from __future__ import annotations

import json
import socket
import socketserver
import threading
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

import pytest

LOOPBACK_NAMES = frozenset({"127.0.0.1", "localhost", "::1"})
MODELS_PATH = "/v1/models"


@dataclass(frozen=True)
class Reply:
    """One canned response: status, raw body bytes, content type, a delay, and the options below.

    `headers` are extra reply headers. `echo` is (marker, request header name):
    every occurrence of the marker in the body is replaced by the value of that
    header as received. `raw_response`, when set, is written as the whole
    response instead (status line included), for replies that are not valid
    HTTP; its markers are replaced too.
    """

    status: int = 200
    body: bytes = b""
    content_type: str = "application/json"
    delay_s: float = 0.0
    headers: tuple[tuple[str, str], ...] = ()
    echo: tuple[str, str] | None = None
    raw_response: bytes | None = None


@dataclass(frozen=True)
class Seen:
    """One request the stub received; header names are lowercased and `body` is the parsed JSON or None."""

    method: str
    target: str
    path: str
    headers: Mapping[str, str]
    raw_body: bytes
    body: Any

    def header(self, name: str) -> str | None:
        """Return the value of a request header by case-insensitive name, or None when it was not sent."""
        return self.headers.get(name.lower())


NOT_FOUND = Reply(status=404, body=b'{"error": "no stub route"}')


class _Server(ThreadingHTTPServer):
    """A threading HTTP server that skips the reverse name lookup HTTPServer does when it binds."""

    daemon_threads = True

    def server_bind(self) -> None:
        """Bind the socket and record the name and port without calling socket.getfqdn."""
        socketserver.TCPServer.server_bind(self)
        self.server_name = "127.0.0.1"
        self.server_port = self.server_address[1]


class StubServer:
    """A local HTTP stub: canned replies by (method, path) and a record of every request."""

    def __init__(self) -> None:
        """Start serving on 127.0.0.1 with an OS-chosen port in a daemon thread."""
        self._routes: dict[tuple[str, str], Reply] = {}
        self._requests: list[Seen] = []
        self._lock = threading.Lock()
        self.release = threading.Event()
        self._server = _Server(("127.0.0.1", 0), _handler_class(self))
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        self._thread.start()

    @property
    def port(self) -> int:
        """Return the port the stub listens on."""
        return int(self._server.server_address[1])

    @property
    def url(self) -> str:
        """Return the stub's base URL, `http://127.0.0.1:<port>`, with no trailing slash."""
        return f"http://127.0.0.1:{self.port}"

    def reply(
        self,
        method: str,
        path: str,
        *,
        status: int = 200,
        json_body: Any = None,
        raw: bytes | None = None,
        content_type: str = "application/json",
        delay_s: float = 0.0,
        headers: Mapping[str, str] | None = None,
        echo: tuple[str, str] | None = None,
        raw_response: bytes | None = None,
    ) -> None:
        """Answer `method path` with `raw` bytes, or else `json_body` encoded as JSON (None means an empty body).

        See Reply for `headers`, `echo`, and `raw_response`.
        """
        body = raw if raw is not None else (b"" if json_body is None else json.dumps(json_body).encode("utf-8"))
        extra = tuple((headers or {}).items())
        with self._lock:
            self._routes[(method, path)] = Reply(status, body, content_type, delay_s, extra, echo, raw_response)

    def serve_models(self, entries: Sequence[str | Mapping[str, Any]], path: str = MODELS_PATH) -> None:
        """Serve an OpenAI-style model list at `path`; a plain string becomes `{"id": <it>, "object": "model"}`."""
        data = [{"id": entry, "object": "model"} if isinstance(entry, str) else dict(entry) for entry in entries]
        self.reply("GET", path, json_body={"object": "list", "data": data})

    @property
    def requests(self) -> list[Seen]:
        """Return every request received so far, in arrival order."""
        with self._lock:
            return list(self._requests)

    def calls(self) -> list[tuple[str, str]]:
        """Return (method, path) of every request received so far, in arrival order."""
        return [(seen.method, seen.path) for seen in self.requests]

    def requests_to(self, method: str, path: str) -> list[Seen]:
        """Return the requests received for `method path`, in arrival order."""
        return [seen for seen in self.requests if (seen.method, seen.path) == (method, path)]

    def _answer(self, seen: Seen) -> Reply:
        """Record a request and return the reply configured for it."""
        with self._lock:
            self._requests.append(seen)
            return self._routes.get((seen.method, seen.path), NOT_FOUND)

    def close(self) -> None:
        """Wake any delayed reply, stop serving, and close the listening socket."""
        self.release.set()
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


def _parse_json(raw: bytes) -> Any:
    """Return the JSON value of a request body, or None when it is empty or not JSON."""
    if not raw:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except ValueError:
        return None


def _echoed(data: bytes, echo: tuple[str, str] | None, headers: Mapping[str, str]) -> bytes:
    """Return `data` with the echo marker replaced by the named request header's value (lowercased names)."""
    if echo is None:
        return data
    marker, name = echo
    return data.replace(marker.encode("ascii"), headers.get(name.lower(), "").encode("latin-1"))


def _handler_class(stub: StubServer) -> type[BaseHTTPRequestHandler]:
    """Return a request handler class bound to `stub`."""

    class Handler(BaseHTTPRequestHandler):
        """Records each request in the stub and writes the configured reply."""

        def _handle(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            headers = {name.lower(): value for name, value in self.headers.items()}
            path = urlsplit(self.path).path
            seen = Seen(self.command, self.path, path, headers, raw, _parse_json(raw))
            reply = stub._answer(seen)
            if reply.delay_s:
                stub.release.wait(reply.delay_s)
            try:
                if reply.raw_response is not None:
                    self.wfile.write(_echoed(reply.raw_response, reply.echo, headers))
                    return
                body = _echoed(reply.body, reply.echo, headers)
                self.send_response(reply.status)
                self.send_header("Content-Type", reply.content_type)
                self.send_header("Content-Length", str(len(body)))
                for name, value in reply.headers:
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(body)
            except OSError:
                pass  # The client gave up (for example after its timeout); nothing to answer.

        do_GET = do_POST = do_PUT = do_DELETE = _handle

        def log_message(self, format: str, *args: Any) -> None:
            """Keep the test output quiet."""

    return Handler


@pytest.fixture
def stub_server() -> Iterator[StubServer]:
    """Run a StubServer for one test and shut it down afterwards."""
    stub = StubServer()
    try:
        yield stub
    finally:
        stub.close()


@pytest.fixture
def dead_port() -> int:
    """Return a 127.0.0.1 port that had a listener a moment ago and has none now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse name lookups of every host except the loopback names, so no test can reach the network."""
    real_getaddrinfo = socket.getaddrinfo
    real_gethostbyname = socket.gethostbyname

    def allowed(host: Any) -> bool:
        name = host.decode("ascii", "replace") if isinstance(host, bytes) else host
        return name is None or name in LOOPBACK_NAMES

    def getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        if not allowed(host):
            raise socket.gaierror(f"test network guard: lookup of {host!r} refused")
        return real_getaddrinfo(host, *args, **kwargs)

    def gethostbyname(host: Any) -> str:
        if not allowed(host):
            raise socket.gaierror(f"test network guard: lookup of {host!r} refused")
        return real_gethostbyname(host)

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr(socket, "gethostbyname", gethostbyname)
