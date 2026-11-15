from __future__ import annotations

import json
import shutil
import tempfile
import unittest
import wave
from pathlib import Path

from bcsubts.cli import main
from bcsubts.wavio import write_wav16
import numpy as np


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg tools are not on PATH")
class CliErrorTests(unittest.TestCase):
    def _valid_inputs(self, root: Path) -> tuple[Path, Path, Path]:
        audio = root / "audio.wav"
        samples = np.zeros((24000, 1), dtype=np.float64)
        write_wav16(audio, samples, 48000)
        subtitles = root / "subs.srt"
        subtitles.write_text(
            "1\n00:00:00,000 --> 00:00:00,400\nHello\n",
            encoding="utf-8",
        )
        events = root / "events.json"
        events.write_text(
            json.dumps({"events": [{"event_id": 1, "type": "splice_null", "offset_ms": 0}]}),
            encoding="utf-8",
        )
        return audio, subtitles, events

    def test_exit_codes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            audio, subtitles, events = self._valid_inputs(root)
            bad_audio = root / "bad24.wav"
            with wave.open(str(bad_audio), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(3)
                wav.setframerate(48000)
                wav.writeframes(b"\x00\x00\x00" * 240)
            code = main(
                [
                    "run",
                    "--audio",
                    str(bad_audio),
                    "--subtitles",
                    str(subtitles),
                    "--scte35",
                    str(events),
                    "--output-dir",
                    str(root / "format"),
                ]
            )
            self.assertEqual(code, 2)

            bad_srt = root / "bad.srt"
            bad_srt.write_text("1\n00:00:75,000 --> 00:00:76,000\nBad\n", encoding="utf-8")
            code = main(
                [
                    "run",
                    "--audio",
                    str(audio),
                    "--subtitles",
                    str(bad_srt),
                    "--scte35",
                    str(events),
                    "--output-dir",
                    str(root / "timecode"),
                ]
            )
            self.assertEqual(code, 3)

            overflow = root / "overflow.json"
            overflow.write_text(
                json.dumps({"events": [{"event_id": 1, "type": "splice_null", "offset_ms": 99999}]}),
                encoding="utf-8",
            )
            code = main(
                [
                    "run",
                    "--audio",
                    str(audio),
                    "--subtitles",
                    str(subtitles),
                    "--scte35",
                    str(overflow),
                    "--output-dir",
                    str(root / "overflow"),
                ]
            )
            self.assertEqual(code, 4)


if __name__ == "__main__":
    unittest.main()
