import unittest
from hashlib import sha256

from snapshot_crawler.normalize import (
    StreamingHasher,
    normalize_text,
    parse_content_type,
)


class NormalizeTests(unittest.TestCase):
    def test_line_endings_and_trailing_whitespace_collapse(self):
        a = "line one   \n   \nline two\t\n"
        b = "line one\r\n\r\nline two\r\n"
        self.assertEqual(normalize_text(a), normalize_text(b))

    def test_streaming_matches_reference_hash(self):
        text = "alpha\r\nbravo   \n\t\ncharlie\n"
        normalized = normalize_text(text)
        expected = sha256(normalized.encode("utf-8")).hexdigest()

        for chunk_size in (1, 3, 7, 4096):
            hasher = StreamingHasher(True, "utf-8", chunk_size=chunk_size)
            raw = text.encode("utf-8")
            for i in range(0, len(raw), chunk_size):
                hasher.update(raw[i : i + chunk_size])
            digest, failed, enc = hasher.finish()
            self.assertEqual(digest, expected)
            self.assertFalse(failed)
            self.assertEqual(enc, "utf-8")

    def test_only_whitespace_differences_share_hash(self):
        lf = "<html>\n<body>\nhello world\n</body>\n</html>"
        crlf_tail = "<html>  \r\n<body>\t\r\nhello world   \r\n</body>\r\n</html>"
        h1 = StreamingHasher(True, "utf-8")
        h2 = StreamingHasher(True, "utf-8")
        h1.update(lf.encode())
        h2.update(crlf_tail.encode())
        self.assertEqual(h1.finish()[0], h2.finish()[0])

    def test_invalid_utf8_falls_back_to_raw_bytes(self):
        raw = b"<html>bad \xff\xfe bytes</html>"
        hasher = StreamingHasher(True, "utf-8")
        hasher.update(raw)
        digest, failed, enc = hasher.finish()
        self.assertTrue(failed)
        self.assertEqual(enc, "raw-bytes")
        self.assertEqual(digest, sha256(raw).hexdigest())

    def test_non_text_hashes_raw_bytes(self):
        raw = bytes(range(256)) * 4
        hasher = StreamingHasher(False, None)
        for i in range(0, len(raw), 17):
            hasher.update(raw[i : i + 17])
        digest, failed, enc = hasher.finish()
        self.assertEqual(digest, sha256(raw).hexdigest())
        self.assertFalse(failed)

    def test_latin1_charset_decode(self):
        raw = "caf\xe9".encode("latin-1")
        hasher = StreamingHasher(True, "iso-8859-1")
        hasher.update(raw)
        digest, failed, enc = hasher.finish()
        self.assertFalse(failed)
        self.assertEqual(enc, "iso-8859-1")
        expected = sha256("caf\xe9".encode("utf-8")).hexdigest()
        self.assertEqual(digest, expected)

    def test_parse_content_type(self):
        is_text, charset = parse_content_type("text/html; charset=gbk")
        self.assertTrue(is_text)
        self.assertEqual(charset, "gbk")
        is_text, charset = parse_content_type("application/octet-stream")
        self.assertFalse(is_text)
        self.assertIsNone(charset)
        is_text, _ = parse_content_type(None)
        self.assertFalse(is_text)

    def test_unknown_charset_falls_back_to_utf8(self):
        hasher = StreamingHasher(True, "x-no-such-codec")
        hasher.update("héllo".encode("utf-8"))
        digest, failed, enc = hasher.finish()
        self.assertFalse(failed)
        self.assertEqual(enc, "utf-8")
        self.assertEqual(
            digest, sha256("héllo".encode("utf-8")).hexdigest()
        )

    def test_oversized_text_releases_raw_buffer(self):
        hasher = StreamingHasher(True, "utf-8", max_bytes=16)
        payload = ("x" * 100 + "\n").encode("utf-8")
        hasher.update(payload)
        digest, failed, enc = hasher.finish()
        self.assertEqual(enc, "raw-bytes")
        self.assertEqual(len(hasher._raw), 0)
        self.assertEqual(digest, sha256(payload).hexdigest())


if __name__ == "__main__":
    unittest.main()
