"""Deterministic local test web servers (standard library only).

Two ThreadingHTTPServer instances on loopback ephemeral ports simulate two
independent origins. Bodies come from fixed seeded data so repeated runs are
byte-identical. Every request is recorded (arrival time, origin, path) for
acceptance checks. No external network is ever contacted.
"""
from __future__ import annotations

import json
import random
import socket
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

SEED_BODY_A = (
    "<html><head><title>Alpha</title></head><body>\n"
    "<h1>Alpha root</h1>\r\n"
    "<p>Same content as beta /shared but different line endings later.</p>\n"
    '<a href="/page1">page 1</a> '
    '<a href="/page2">page 2</a> '
    '<a href="/shared">shared</a> '
    '<a href="/slow">slow</a> '
    '<a href="/flaky">flaky</a> '
    '<a href="/always500">always 500</a> '
    '<a href="/missing">missing</a> '
    '<a href="/reset">reset</a> '
    '<a href="/oversized">oversized</a> '
    '<a href="/etag">etag</a> '
    '<a href="/versioned">versioned</a> '
    '<a href="/external-redirect">external redirect</a> '
    '<a href="/query?a=1&b=2#frag">query with fragment</a>'
    "</body></html>"
)

PAGE1 = (
    "<html><body><h1>Page 1</h1>\n"
    '<a href="/page2">page 2</a> '
    '<a href="/deep/1">deep 1</a></body></html>'
)
PAGE2 = (
    "<html><body><h1>Page 2</h1>\r\n"
    '<a href="/deep/1">deep 1</a></body></html>'
)
DEEP1 = (
    "<html><body><h1>Deep 1</h1>"
    '<a href="/deep/2">deep 2</a></body></html>'
)
DEEP2 = "<html><body><h1>Deep 2</h1></body></html>"

# The same logical text with CRLF + trailing whitespace: must hash identically
# to the LF version after normalization.
SHARED_LF = (
    "<html><body><h1>Shared document</h1>\n"
    "line one   \n"
    "   \n"
    "line two\t\n"
    "</body></html>"
)
SHARED_CRLF = SHARED_LF.replace("\n", "\r\n")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


class RequestLog:
    def __init__(self):
        self.lock = threading.Lock()
        self.entries: list[dict] = []

    def add(self, origin: str, path: str, client: str):
        mono = time.monotonic()
        with self.lock:
            entry = {
                "arrived_at": _now(),
                "mono": mono,
                "completed_mono": None,
                "origin": origin,
                "path": path,
                "client": client,
            }
            self.entries.append(entry)
            return entry

    def complete(self, entry):
        entry["completed_mono"] = time.monotonic()


def make_handler(origin_label: str, log: RequestLog, state_dir: Path, token: str,
                 state_key: str = "default"):
    rng = random.Random(20261113)
    big_chunk = bytes(rng.getrandbits(8) for _ in range(64 * 1024))

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            return

        def _send(self, status: int, body: bytes, content_type="text/html; charset=utf-8",
                  extra_headers=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Last-Modified", "Wed, 01 Jan 2025 00:00:00 GMT")
            if extra_headers:
                for key, value in extra_headers.items():
                    self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urlsplit(self.path)
            path = parsed.path
            entry = log.add(origin_label, self.path, self.client_address[0])
            try:
                self.route(path, parsed.query)
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                log.complete(entry)

        def route(self, path: str, query: str):
            if path == "/":
                self._send(200, SEED_BODY_A.encode("utf-8"))
            elif path == "/page1":
                self._send(200, PAGE1.encode("utf-8"))
            elif path == "/page2":
                self._send(200, PAGE2.encode("utf-8"))
            elif path == "/deep/1":
                self._send(200, DEEP1.encode("utf-8"))
            elif path == "/deep/2":
                self._send(200, DEEP2.encode("utf-8"))
            elif path == "/shared":
                # CRLF + trailing whitespace variant.
                self._send(200, SHARED_CRLF.encode("utf-8"))
            elif path == "/query":
                body = f"<html><body>query={query}</body></html>".encode()
                self._send(200, body)
            elif path == "/slow":
                time.sleep(1.5)
                self._send(200, b"<html><body>slow</body></html>")
            elif path == "/flaky":
                # Key includes the local port so two origins have independent
                # failure sequences even when crawled concurrently. Each origin
                # deterministically fails the first two requests, then recovers.
                flag = state_dir / (
                    f"flaky-{token}-{state_key}-{self.server.server_address[1]}.count"
                )
                count = int(flag.read_text()) if flag.exists() else 0
                count += 1
                flag.write_text(str(count))
                if count <= 2:
                    self._send(503, b"transient failure")
                else:
                    self._send(200, b"<html><body>flaky recovered</body></html>")
            elif path == "/always500":
                self._send(500, b"permanent server error")
            elif path == "/missing":
                self._send(404, b"not found")
            elif path == "/bad":
                self._send(400, b"bad request")
            elif path == "/reset":
                # Abrupt TCP reset with no HTTP response.
                self.close_connection = True
                raw = self.connection
                raw.setsockopt(
                    socket.SOL_SOCKET, socket.SO_LINGER,
                    _linger_struct(),
                )
                raw.close()
            elif path == "/oversized":
                total_size = 200 * 64 * 1024
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(total_size))
                self.send_header("Connection", "close")
                self.end_headers()
                for _ in range(200):  # 200 * 64KiB = ~12.5 MiB
                    self.wfile.write(big_chunk)
                self.wfile.flush()
            elif path == "/etag":
                etag = '"v1-fixed-etag"'
                if self.headers.get("If-None-Match") == etag:
                    self.send_response(304)
                    self.send_header("ETag", etag)
                    self.end_headers()
                else:
                    self._send(
                        200,
                        b"<html><body>etagged</body></html>",
                        extra_headers={"ETag": etag},
                    )
            elif path == "/versioned":
                flag = state_dir / (
                    f"versioned-{token}-{state_key}-"
                    f"{self.server.server_address[1]}.count"
                )
                count = int(flag.read_text()) if flag.exists() else 0
                count += 1
                flag.write_text(str(count))
                body = f"<html><body>revision {count}</body></html>".encode()
                self._send(200, body)
            elif path == "/redirect-internal":
                self.send_response(302)
                self.send_header("Location", "/page1")
                self.send_header("Content-Length", "0")
                self.end_headers()
            elif path == "/external-redirect":
                self.send_response(302)
                self.send_header("Location", "http://example.invalid/elsewhere")
                self.send_header("Content-Length", "0")
                self.end_headers()
            elif path == "/latin1":
                # Declared latin-1 body with valid latin-1 bytes.
                self._send(
                    200,
                    "<html><body>caf\xe9 \xf1</body></html>".encode("latin-1"),
                    content_type="text/html; charset=iso-8859-1",
                )
            elif path == "/bad-charset":
                # Content claims UTF-8 but contains invalid UTF-8 bytes.
                self._send(
                    200,
                    b"<html><body>bad \xff\xfe bytes</body></html>",
                    content_type="text/html; charset=utf-8",
                )
            else:
                self._send(404, b"unknown route")

    return Handler


def _linger_struct():
    import struct

    # onoff=1, linger=0 -> RST on close instead of FIN.
    return struct.pack("ii", 1, 0)


class TestServer:
    def __init__(self, label: str, log: RequestLog, state_dir: Path, token: str,
                 state_key: str = "default"):
        self.label = label
        self.httpd = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            make_handler(label, log, state_dir, token, state_key),
        )
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(
            target=self.httpd.serve_forever, daemon=True
        )

    @property
    def root_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/"

    def start(self):
        self.thread.start()

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def start_pair(state_dir: Path, token: str, state_key: str = "default"):
    log = RequestLog()
    servers = [
        TestServer("alpha", log, state_dir, token, state_key),
        TestServer("beta", log, state_dir, token, state_key),
    ]
    for server in servers:
        server.start()
    return servers, log


if __name__ == "__main__":
    import tempfile

    state = Path(tempfile.mkdtemp())
    servers, log = start_pair(state, "manual")
    print(json.dumps({s.label: s.root_url for s in servers}, indent=2))
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        for server in servers:
            server.stop()
