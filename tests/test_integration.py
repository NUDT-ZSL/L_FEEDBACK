import json
import os
import subprocess
import sys
import tempfile
import unittest
import wave

import numpy as np

from audiopipeline.config import SAMPLE_RATE
from audiopipeline.decoder import decode_audio
from audiopipeline.errors import PipelineError
from audiopipeline.writer import write_wav24


def write_wav(path, freq=440, seconds=1.0, rate=44100, channels=1, amp=0.5):
    t = np.arange(int(seconds * rate)) / rate
    sig = amp * np.sin(2 * np.pi * freq * t)
    audio = np.tile(sig[:, None], (1, channels)).astype(np.float32)
    pcm = np.rint(audio * 32767).astype("<i2")
    with wave.open(path, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm.tobytes())


class DecoderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_resample_mono_to_48k_stereo(self):
        path = os.path.join(self.tmp.name, "mono.wav")
        write_wav(path, rate=22050, channels=1, seconds=1.0)
        audio, rate = decode_audio(path)
        self.assertEqual(rate, SAMPLE_RATE)
        self.assertEqual(audio.shape[1], 2)
        self.assertEqual(audio.shape[0], SAMPLE_RATE)
        self.assertTrue(np.isfinite(audio).all())

    def test_corrupt_file_raises_decode_error(self):
        path = os.path.join(self.tmp.name, "bad.wav")
        with open(path, "wb") as fh:
            fh.write(b"RIFF" + bytes(128))
        with self.assertRaises(PipelineError) as ctx:
            decode_audio(path)
        self.assertEqual(ctx.exception.code, "E_DECODE_FAILED")


class WriterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_24bit_wav_roundtrip_no_clipping(self):
        path = os.path.join(self.tmp.name, "out.wav")
        audio = np.linspace(-0.9, 0.9, SAMPLE_RATE, dtype=np.float32)
        stereo = np.tile(audio[:, None], (1, 2))
        write_wav24(path, stereo)
        with wave.open(path, "rb") as wav:
            self.assertEqual(wav.getframerate(), SAMPLE_RATE)
            self.assertEqual(wav.getsampwidth(), 3)
            self.assertEqual(wav.getnchannels(), 2)
            raw = wav.readframes(wav.getnframes())
        data = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        unsigned = (
            data[:, 0].astype(np.uint32)
            | (data[:, 1].astype(np.uint32) << 8)
            | (data[:, 2].astype(np.uint32) << 16)
        )
        ints = (unsigned.astype(np.int32) << 8) >> 8  # sign-extend 24-bit
        self.assertLessEqual(np.abs(ints).max(), 2**23 - 1)
        # round-trip amplitude stays within one quantization step
        recovered = (ints / (2**23 - 1)).reshape(-1, 2)
        # One 24-bit quantization step is ~1.19e-7 full scale.
        self.assertLess(np.max(np.abs(recovered - stereo)), 1.3e-7)


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        write_wav(os.path.join(self.root, "a.wav"), rate=48000,
                  channels=2, seconds=2.0)

    def tearDown(self):
        self.tmp.cleanup()

    def _run_cli(self, manifest_name="manifest.json"):
        proc = subprocess.run(
            [sys.executable, "-m", "audiopipeline.cli",
             os.path.join(self.root, manifest_name),
             os.path.join(self.root, "out.wav"),
             "-r", os.path.join(self.root, "report.json")],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            capture_output=True, text=True,
        )
        return proc

    def test_success_and_report(self):
        manifest = {
            "total_duration_ms": 2000,
            "events": [
                {"id": "e1", "path": "a.wav", "start_ms": 0,
                 "category": "dialog", "duration_ms": 1500, "priority": 1}
            ],
        }
        with open(os.path.join(self.root, "manifest.json"), "w") as fh:
            json.dump(manifest, fh)
        proc = self._run_cli()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        report = json.load(open(os.path.join(self.root, "report.json")))
        self.assertEqual(report["summary"]["succeeded"], 1)
        event = report["events"][0]
        self.assertEqual(event["actual_start_ms"], 0.0)
        self.assertEqual(event["actual_end_ms"], 1500.0)
        self.assertTrue(event["trimmed"])

    def test_fatal_manifest_nonzero_exit(self):
        with open(os.path.join(self.root, "manifest.json"), "w") as fh:
            json.dump({"total_duration_ms": -1, "events": []}, fh)
        proc = self._run_cli()
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("E_DURATION_INVALID", proc.stderr)


if __name__ == "__main__":
    unittest.main()
