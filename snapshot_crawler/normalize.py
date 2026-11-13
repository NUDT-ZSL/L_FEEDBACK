"""Content normalization and SHA-256 hashing with streaming support.

Text/html (and other text/*) bodies are decoded with the declared charset
(UTF-8 default). Line endings are normalized to LF, trailing whitespace is
removed from every line, and pure-whitespace lines collapse to empty lines.
The normalized text is re-encoded as UTF-8 before hashing. Non-text payloads
are hashed as raw bytes. Everything is processed in fixed-size chunks so an
oversized body is never fully held in memory.
"""
from __future__ import annotations

import hashlib


def normalize_text(text: str) -> str:
    """Pure function used by unit tests and the streaming hasher."""
    out_lines = []
    # Universal newline handling: splitlines covers \r\n, \r and \n.
    for line in text.splitlines():
        stripped = line.rstrip()
        out_lines.append("" if not stripped.strip() else stripped)
    return "\n".join(out_lines)


class StreamingHasher:
    """Incrementally consumes raw response bytes and produces a content hash.

    Text is decoded incrementally with the declared charset. A one-line tail
    buffer keeps chunks split mid-line correct. The raw bytes of text bodies
    are retained (bounded by the configured max response size) so that a
    decode failure can fall back to hashing the original bytes as required.
    Non-text payloads never retain more than one chunk in memory.
    """

    def __init__(
        self,
        is_text: bool,
        charset: str | None,
        chunk_size: int = 65536,
        max_bytes: int | None = None,
    ):
        import codecs

        self.is_text = is_text
        self.chunk_size = chunk_size
        self.sha = hashlib.sha256()
        self._decoder = None
        self._tail = ""
        self._raw = bytearray()
        self._failed = False
        self._max_bytes = max_bytes
        self._lines_emitted = 0
        self._pending_line: str | None = None
        self.encoding_used = "raw-bytes"
        self.encoding_failed = False
        if is_text:
            enc = (charset or "utf-8").strip() or "utf-8"
            try:
                codecs.lookup(enc)
            except LookupError:
                enc = "utf-8"
            self._decoder = codecs.getincrementaldecoder(enc)(errors="strict")
            self.encoding_used = enc

    def _switch_to_raw(self) -> None:
        self._failed = True
        self.encoding_failed = True
        self.encoding_used = "raw-bytes"
        self.sha = hashlib.sha256()
        self.sha.update(bytes(self._raw))
        self._raw = bytearray()  # release the fallback buffer
        self._tail = ""

    def _emit(self, text: str) -> None:
        if not text:
            return
        buf = self._tail + text
        self._tail = ""
        start = 0
        index = 0
        while index < len(buf):
            char = buf[index]
            if char not in ("\r", "\n"):
                index += 1
                continue
            line_body = buf[start:index]
            index += 1
            if char == "\r":
                if index < len(buf):
                    if buf[index] == "\n":
                        index += 1
                else:
                    # Bare CR at chunk edge: wait for the next chunk to learn
                    # whether a LF follows (CRLF) or this is a lone CR.
                    self._tail = buf[start:]
                    return
            # Flush any line deferred from an earlier newline, then defer this
            # one: a deferred final line means a trailing empty line, which is
            # dropped in finish(), matching str.splitlines().
            if self._pending_line is not None:
                self._hash_line(self._pending_line)
            self._pending_line = line_body
            start = index
        self._tail = buf[start:]

    def _hash_line(self, body: str) -> None:
        stripped = body.rstrip()
        if not stripped.strip():
            stripped = ""
        if self._lines_emitted:
            self.sha.update(b"\n")
        self.sha.update(stripped.encode("utf-8"))
        self._lines_emitted += 1

    def update(self, data: bytes) -> None:
        if self._failed or not self.is_text:
            self.sha.update(data)
            return
        self._raw.extend(data)
        if self._max_bytes is not None and len(self._raw) > self._max_bytes:
            # Bounded fallback: an oversized text body cannot keep the whole
            # payload buffered; _switch_to_raw already hashes the retained
            # bytes (including this chunk), so just release the buffer.
            self._switch_to_raw()
            return
        try:
            text = self._decoder.decode(data, final=False)
        except UnicodeDecodeError:
            self._switch_to_raw()
            return
        self._emit(text)

    def finish(self) -> tuple[str, bool, str]:
        if self.is_text and not self._failed:
            tail = self._decoder.decode(b"", final=True)
            self._tail += tail
            if self._tail:
                stripped = self._tail.rstrip()
                if self._pending_line is not None:
                    self._hash_line(self._pending_line)
                    self._pending_line = None
                if self._lines_emitted:
                    self.sha.update(b"\n")
                self.sha.update(stripped.encode("utf-8"))
            elif self._pending_line is not None:
                # Input ended with a newline: the deferred line is the final
                # real line, not an extra empty one.
                self._hash_line(self._pending_line)
                self._pending_line = None
        return self.sha.hexdigest(), self.encoding_failed, self.encoding_used


def parse_content_type(content_type: str | None) -> tuple[bool, str | None]:
    """Return (is_text, charset) from an HTTP Content-Type header."""
    if not content_type:
        return False, None
    header = content_type.lower()
    media = header.split(";", 1)[0].strip()
    is_text = media == "text/html" or media.startswith("text/") or (
        media.endswith("+xml") or media in ("application/xhtml+xml", "application/xml")
    )
    charset = None
    for param in header.split(";")[1:]:
        param = param.strip()
        if param.startswith("charset="):
            charset = param.split("=", 1)[1].strip().strip('"')
    return is_text, charset
