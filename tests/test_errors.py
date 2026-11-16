import json
import os
import stat
import unittest

import numpy as np
import soundfile as sf

from rhythm_mixer.engine import run
from rhythm_mixer.errors import (
    BadWavError,
    ChartError,
    EventError,
    GainError,
    NotOnBeatError,
    UnknownLayerError,
)

from tests._helpers import make_fixture, read_json, run_cli, write_json, write_wav


class ErrorConditionTests(unittest.TestCase):
    def _fixture(self):
        input_dir, output_dir = make_fixture()
        return input_dir, output_dir

    def _chart(self, input_dir):
        return read_json(os.path.join(input_dir, "chart.json"))

    def _events(self, input_dir):
        return read_json(os.path.join(input_dir, "events.json"))

    def test_unknown_layer(self):
        input_dir, output_dir = self._fixture()
        events = self._events(input_dir)
        events["events"].append(
            {"time_ms": 0.0, "layer": "kazoo", "action": "enable"}
        )
        write_json(os.path.join(input_dir, "events.json"), events)
        with self.assertRaises(UnknownLayerError):
            run(input_dir, output_dir)

    def test_non_finite_gain(self):
        input_dir, output_dir = self._fixture()
        events = self._events(input_dir)
        events["events"][0]["gain"] = float("nan")
        write_json(os.path.join(input_dir, "events.json"), events)
        with self.assertRaises(GainError):
            run(input_dir, output_dir)

        input_dir2, output_dir2 = self._fixture()
        events2 = self._events(input_dir2)
        events2["events"][0]["gain"] = float("inf")
        write_json(os.path.join(input_dir2, "events.json"), events2)
        with self.assertRaises(GainError):
            run(input_dir2, output_dir2)

    def test_event_not_on_beat(self):
        input_dir, output_dir = self._fixture()
        events = self._events(input_dir)
        events["events"][0]["time_ms"] = 123.0
        write_json(os.path.join(input_dir, "events.json"), events)
        with self.assertRaises(NotOnBeatError):
            run(input_dir, output_dir)

    def test_negative_event_time(self):
        input_dir, output_dir = self._fixture()
        events = self._events(input_dir)
        events["events"][0]["time_ms"] = -1.0
        write_json(os.path.join(input_dir, "events.json"), events)
        with self.assertRaises(EventError):
            run(input_dir, output_dir)

    def test_invalid_bpm(self):
        input_dir, output_dir = self._fixture()
        chart = self._chart(input_dir)
        chart["bpm"] = 0
        write_json(os.path.join(input_dir, "chart.json"), chart)
        with self.assertRaises(ChartError):
            run(input_dir, output_dir)

    def test_bad_wav_header(self):
        input_dir, output_dir = self._fixture()
        path = os.path.join(input_dir, "drums.wav")
        with open(path, "wb") as handle:
            handle.write(b"NOTRIFF" + os.urandom(64))
        with self.assertRaises(BadWavError):
            run(input_dir, output_dir)

    def test_sample_rate_mismatch(self):
        input_dir, output_dir = self._fixture()
        path = os.path.join(input_dir, "drums.wav")
        samples = np.zeros(512, dtype=np.float32)
        write_wav(path, samples, samplerate=22050)
        from rhythm_mixer.errors import SampleRateError
        with self.assertRaises(SampleRateError):
            run(input_dir, output_dir)

    def test_stereo_wav_rejected(self):
        input_dir, output_dir = self._fixture()
        path = os.path.join(input_dir, "drums.wav")
        stereo = np.zeros((512, 2), dtype=np.float32)
        sf.write(path, stereo, 44100, subtype="PCM_16")
        with self.assertRaises(BadWavError):
            run(input_dir, output_dir)

    def test_non_16bit_wav_rejected(self):
        input_dir, output_dir = self._fixture()
        path = os.path.join(input_dir, "drums.wav")
        samples = np.zeros(512, dtype=np.float32)
        write_wav(path, samples, samplerate=44100, subtype="FLOAT")
        with self.assertRaises(BadWavError):
            run(input_dir, output_dir)

    def test_missing_layer_file(self):
        input_dir, output_dir = self._fixture()
        os.remove(os.path.join(input_dir, "bass.wav"))
        with self.assertRaises(BadWavError):
            run(input_dir, output_dir)

    def test_malformed_json(self):
        input_dir, output_dir = self._fixture()
        with open(os.path.join(input_dir, "chart.json"), "w") as handle:
            handle.write("{not valid json")
        with self.assertRaises(ChartError):
            run(input_dir, output_dir)


class CliErrorTests(unittest.TestCase):
    def test_stderr_json_and_exit_codes(self):
        input_dir, output_dir = make_fixture()
        events = read_json(os.path.join(input_dir, "events.json"))
        events["events"].append(
            {"time_ms": 0.0, "layer": "ghost", "action": "enable"}
        )
        write_json(os.path.join(input_dir, "events.json"), events)

        result = run_cli(input_dir, output_dir)
        self.assertEqual(result.returncode, 1)
        payload = json.loads(result.stderr.strip())
        self.assertEqual(payload["error"], "E_UNKNOWN_LAYER")
        self.assertIn("ghost", payload["message"])

    def test_bad_wav_exit_code_2(self):
        input_dir, output_dir = make_fixture()
        with open(os.path.join(input_dir, "drums.wav"), "wb") as handle:
            handle.write(b"XXXX" + os.urandom(128))
        result = run_cli(input_dir, output_dir)
        self.assertEqual(result.returncode, 2)
        payload = json.loads(result.stderr.strip())
        self.assertEqual(payload["error"], "E_BAD_WAV")

    def test_unwritable_output_dir_cleans_partials(self):
        if os.geteuid() == 0:
            self.skipTest("root bypasses directory write permissions")
        input_dir, output_dir = make_fixture()
        os.makedirs(output_dir)
        os.chmod(output_dir, stat.S_IREAD | stat.S_IXUSR)
        try:
            result = run_cli(input_dir, output_dir)
            self.assertEqual(result.returncode, 2)
            payload = json.loads(result.stderr.strip())
            self.assertEqual(payload["error"], "E_OUTPUT")
        finally:
            os.chmod(output_dir, stat.S_IRWXU)
        self.assertFalse(
            os.path.exists(os.path.join(output_dir, "mix_output.wav"))
        )


if __name__ == "__main__":
    unittest.main()
