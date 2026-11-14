import json
import os
import tempfile
import unittest

from subalign.encoding import decode_bytes, detect_encoding
from subalign.subtitles import load_subtitle

from tests.samples import reference_events, write_bytes, write_subtitle


class EncodingTests(unittest.TestCase):
    def test_detects_bom_variants(self) -> None:
        self.assertEqual(detect_encoding(b"\xef\xbb\xbfhello").encoding, "utf-8-sig")
        self.assertEqual(detect_encoding(b"\xff\xfeh\x00").encoding, "utf-16-le")
        self.assertEqual(detect_encoding(b"\xfe\xff\x00h").encoding, "utf-16-be")
        self.assertEqual(detect_encoding("plain".encode()).encoding, "utf-8")

    def test_decode_round_trip(self) -> None:
        text = "字幕 alignment тест"
        for encoding, raw in (
            ("utf-8-sig", b"\xef\xbb\xbf" + text.encode("utf-8")),
            ("utf-16-le", b"\xff\xfe" + text.encode("utf-16-le")),
            ("utf-16-be", b"\xfe\xff" + text.encode("utf-16-be")),
        ):
            decoded, result = decode_bytes(raw)
            self.assertEqual(decoded, text)
            self.assertEqual(result.encoding, encoding)


class SubtitleFormatTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = self.temp.name
        self.events = reference_events(count=8)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_loads_all_supported_encodings(self) -> None:
        for encoding in ("utf-8", "utf-8-bom", "utf-16-le", "utf-16-be"):
            path = os.path.join(self.root, f"sample_{encoding}.srt")
            write_subtitle(path, self.events, "srt", encoding)
            subtitle = load_subtitle(path)
            self.assertEqual(len(subtitle.events), 8)
            self.assertEqual(subtitle.events[0].start, 500)
            self.assertEqual(subtitle.events[3].text, self.events[3][2])

    def test_ass_style_fields_are_preserved(self) -> None:
        path = os.path.join(self.root, "styled.ass")
        write_subtitle(path, self.events, "ass")
        subtitle = load_subtitle(path)
        self.assertEqual(subtitle.fmt, "ass")
        self.assertTrue(all(event.style == "Main" for event in subtitle.events))
        self.assertEqual(subtitle.raw.events[2].style, "Main")

    def test_structured_error_for_missing_file(self) -> None:
        from subalign.subtitles import SubtitleError

        with self.assertRaises(SubtitleError) as context:
            load_subtitle(os.path.join(self.root, "missing.srt"))
        self.assertEqual(context.exception.error_type, "FileNotFound")
        self.assertIn("missing.srt", context.exception.to_dict()["path"])

    def test_structured_error_for_unsupported_extension(self) -> None:
        from subalign.subtitles import SubtitleError

        path = os.path.join(self.root, "notes.txt")
        write_bytes(path, b"hello")
        with self.assertRaises(SubtitleError) as context:
            load_subtitle(path)
        self.assertEqual(context.exception.error_type, "UnsupportedFormat")

    def test_structured_error_for_corrupt_srt(self) -> None:
        from subalign.subtitles import SubtitleError

        path = os.path.join(self.root, "broken.srt")
        write_bytes(path, b"\x80\x81 invalid utf-8 sequence")
        with self.assertRaises(SubtitleError) as context:
            load_subtitle(path)
        self.assertEqual(context.exception.error_type, "EncodingError")
        payload = json.dumps(context.exception.to_dict())
        self.assertIn("error_type", payload)

        truncated = os.path.join(self.root, "truncated.srt")
        write_bytes(truncated, b"\xff\xfe\xd8")
        with self.assertRaises(SubtitleError) as context_two:
            load_subtitle(truncated)
        self.assertEqual(context_two.exception.error_type, "EncodingError")


if __name__ == "__main__":
    unittest.main()
