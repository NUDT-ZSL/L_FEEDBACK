"""HTTP fetching on top of http.client: streaming, redirects, 304, limits.

No third-party libraries and no external network assumptions. Every response
body is read in fixed chunks and streamed to a temporary blob file, so bodies
larger than ``max_bytes`` are truncated without ever being fully buffered.
"""
from __future__ import annotations

import http.client
import os
import socket
from dataclasses import dataclass
from urllib.parse import urlsplit

from .blobstore import BlobStore
from .normalize import StreamingHasher, parse_content_type
from .urlutils import same_origin


class FetchError(Exception):
    """Network/protocol failure that may be retried (or mapped to a type)."""


REDIRECT_STATUS = {301, 302, 303, 307, 308}


@dataclass
class FetchResult:
    url: str
    final_url: str
    http_status: int
    content_hash: str | None
    temp_path: str | None
    size_bytes: int
    oversized: bool
    is_text: bool
    encoding_used: str
    encoding_failed: bool
    last_modified: str | None
    etag: str | None
    not_modified: bool


def _open_connection(url: str, timeout: float):
    parts = urlsplit(url)
    host = parts.hostname
    port = parts.port or (443 if parts.scheme == "https" else 80)
    if parts.scheme == "https":
        conn = http.client.HTTPSConnection(host, port, timeout=timeout)
    else:
        conn = http.client.HTTPConnection(host, port, timeout=timeout)
    return conn, parts


def fetch_url(
    url: str,
    root_url: str,
    blob_store: BlobStore,
    timeout: float,
    max_bytes: int,
    user_agent: str,
    max_redirects: int = 5,
    conditional_headers: dict | None = None,
) -> FetchResult:
    """Fetch one URL, following only same-origin redirects.

    Raises FetchError with a classified message on network failure, cross-origin
    redirect, or too many redirects. HTTP 4xx/5xx are returned as FetchResult.
    """
    current = url
    redirects = 0
    while True:
        try:
            conn, parts = _open_connection(current, timeout)
            path = parts.path or "/"
            if parts.query:
                path = f"{path}?{parts.query}"
            conn.request(
                "GET",
                path,
                headers={
                    "Host": parts.netloc,
                    "User-Agent": user_agent,
                    "Accept": "*/*",
                    "Connection": "close",
                    **(conditional_headers or {}),
                },
            )
            resp = conn.getresponse()
        except (TimeoutError, socket.timeout) as exc:
            raise FetchError("timeout") from exc
        except ConnectionResetError as exc:
            raise FetchError("connection_reset") from exc
        except ConnectionError as exc:
            raise FetchError("connection_error") from exc
        except (http.client.HTTPException, OSError) as exc:
            raise FetchError(f"network_error:{type(exc).__name__}") from exc

        status = resp.status
        if status in REDIRECT_STATUS:
            location = resp.getheader("Location")
            resp.read()
            conn.close()
            if not location:
                raise FetchError("redirect_without_location")
            from .urlutils import normalize_url

            nxt = normalize_url(current, location)
            if nxt is None or not same_origin(nxt, root_url):
                raise FetchError("cross_origin_redirect")
            redirects += 1
            if redirects > max_redirects:
                raise FetchError("too_many_redirects")
            current = nxt
            continue

        headers = {k.lower(): v for k, v in resp.getheaders()}
        last_modified = headers.get("last-modified")
        etag = headers.get("etag")

        if status == 304:
            conn.close()
            return FetchResult(
                url=url,
                final_url=current,
                http_status=304,
                content_hash=None,
                temp_path=None,
                size_bytes=0,
                oversized=False,
                is_text=False,
                encoding_used="",
                encoding_failed=False,
                last_modified=last_modified,
                etag=etag,
                not_modified=True,
            )

        content_type = headers.get("content-type")
        is_text, charset = parse_content_type(content_type)
        hasher = StreamingHasher(is_text, charset, max_bytes=max_bytes)
        fd, temp_name = blob_store.new_temp_file()
        total = 0
        oversized = False
        try:
            with os.fdopen(fd, "wb") as sink:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    if total + len(chunk) > max_bytes:
                        allowed = max_bytes - total
                        if allowed > 0:
                            sink.write(chunk[:allowed])
                            hasher.update(chunk[:allowed])
                            total += allowed
                        oversized = True
                        # Drain and discard the remainder without retaining it.
                        try:
                            while True:
                                rest = resp.read(65536)
                                if not rest:
                                    break
                        except http.client.IncompleteRead:
                            # Connection closed mid-body: expected while
                            # truncating an oversized response.
                            pass
                        break
                    sink.write(chunk)
                    hasher.update(chunk)
                    total += len(chunk)
        finally:
            conn.close()

        content_hash, encoding_failed, encoding_used = hasher.finish()
        return FetchResult(
            url=url,
            final_url=current,
            http_status=status,
            content_hash=content_hash,
            temp_path=temp_name,
            size_bytes=total,
            oversized=oversized,
            is_text=is_text and not encoding_failed,
            encoding_used=encoding_used,
            encoding_failed=encoding_failed,
            last_modified=last_modified,
            etag=etag,
            not_modified=False,
        )
