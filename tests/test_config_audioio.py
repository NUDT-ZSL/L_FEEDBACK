import os
import unittest

import soundfile as sf

from seamless_loop.audioio import to_sample_index
from seamless_loop.config import load_project
from seamless_loop.errors import ConfigError, ParameterRangeError

from helpers import base_project, make_workspace, write_json


class SampleIndexTests(unittest.TestCase):
    def test_half_up_rounding(self):
        # 0.5 samples round up regardless of platform banker's rounding.
        self.assertEqual(to_sample_index(1.5 / 48000, 48000), 2)
        self.assertEqual(to_sample_index(2.5 / 48000, 48000), 3)
        self.assertEqual(to_sample_index(0.25, 48000), 12000)
        self.assertEqual(to_sample_index(1.0, 44100), 44100)

    def test_rejects_non_numeric(self):
        with self.assertRaises(ParameterRangeError):
            to_sample_index("0.25", 48000)


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.root, self.samples_dir, self.output_dir, self.paths = make_workspace()

    def _load(self, payload):
        config_path = write_json(
            os.path.join(self.samples_dir, "project.json"), payload
        )
        return load_project(config_path)

    def test_happy_path_resolves_relative_paths(self):
        project = self._load(base_project(self.samples_dir, self.output_dir))
        self.assertTrue(all(os.path.isabs(p) for p in project["input_files"]))
        self.assertTrue(os.path.isdir(project["output_dir"]))
        self.assertEqual(len(project["input_files"]), 3)

    def test_missing_file_is_config_error(self):
        with self.assertRaises(ConfigError):
            load_project(os.path.join(self.root, "does_not_exist.json"))

    def test_missing_field(self):
        payload = base_project(self.samples_dir, self.output_dir)
        del payload["loop_end"]
        with self.assertRaises(ConfigError):
            self._load(payload)

    def test_invalid_json(self):
        path = os.path.join(self.samples_dir, "broken.json")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("{not valid json")
        with self.assertRaises(ConfigError):
            load_project(path)

    def test_empty_input_list(self):
        payload = base_project(self.samples_dir, self.output_dir, input_files=[])
        with self.assertRaises(ConfigError):
            self._load(payload)

    def test_non_writable_output_dir(self):
        blocked = os.path.join(self.samples_dir, "a_file_not_dir")
        with open(blocked, "w", encoding="utf-8") as handle:
            handle.write("x")
        payload = base_project(self.samples_dir, blocked)
        with self.assertRaises(ConfigError):
            self._load(payload)

    def test_bad_overwrite_type(self):
        payload = base_project(
            self.samples_dir, self.output_dir, overwrite="yes"
        )
        with self.assertRaises(ConfigError):
            self._load(payload)


class RangeConfigTests(unittest.TestCase):
    def setUp(self):
        self.root, self.samples_dir, self.output_dir, self.paths = make_workspace()

    def _load(self, payload):
        config_path = write_json(
            os.path.join(self.samples_dir, "project.json"), payload
        )
        return load_project(config_path)

    def test_negative_fade_rejected(self):
        payload = base_project(
            self.samples_dir, self.output_dir, crossfade_duration=-0.1
        )
        with self.assertRaises(ParameterRangeError):
            self._load(payload)

    def test_zero_fade_rejected(self):
        payload = base_project(
            self.samples_dir, self.output_dir, crossfade_duration=0
        )
        with self.assertRaises(ParameterRangeError):
            self._load(payload)

    def test_loop_end_before_start(self):
        payload = base_project(
            self.samples_dir, self.output_dir,
            loop_start=1.0, loop_end=0.5,
        )
        with self.assertRaises(ParameterRangeError):
            self._load(payload)

    def test_loop_shorter_than_two_fades_is_shortened_per_track(self):
        # A region shorter than two configured fades does not hard-fail; the
        # per-track logic shortens the fade and records the adjustment.
        payload = base_project(
            self.samples_dir, self.output_dir,
            loop_start=0.1, loop_end=0.3, crossfade_duration=0.15,
        )
        project = self._load(payload)
        from seamless_loop.processor import process_project

        output_path = process_project(project)
        report_path = output_path.replace("_loop.wav", "_loop_report.json")
        import json

        with open(report_path, "r", encoding="utf-8") as handle:
            report = json.load(handle)
        # Region = 9600 samples -> max fade 4800; configured fade = 7200.
        for track in report["tracks"]:
            self.assertTrue(track["crossfade_adjusted"])
            self.assertEqual(track["crossfade_samples_used"], 4800)


class AudioValidationTests(unittest.TestCase):
    def setUp(self):
        self.root, self.samples_dir, self.output_dir, self.paths = make_workspace()

    def test_missing_input_raises_file_error(self):
        from seamless_loop.audioio import load_and_validate_track
        from seamless_loop.errors import AudioFileError

        with self.assertRaises(AudioFileError):
            load_and_validate_track(
                os.path.join(self.samples_dir, "nope.wav"), 0, 100
            )

    def test_unsupported_bit_depth(self):
        from seamless_loop.audioio import load_and_validate_track
        from seamless_loop.errors import AudioFileError

        with self.assertRaises(AudioFileError):
            load_and_validate_track(self.paths["unsupported"], 0, 100)

    def test_sample_rate_mismatch(self):
        from seamless_loop.audioio import load_and_validate_track
        from seamless_loop.errors import AudioFileError

        with self.assertRaises(AudioFileError):
            load_and_validate_track(
                self.paths["mismatch"], 0, 100, expected_sr=48000
            )

    def test_region_beyond_duration(self):
        from seamless_loop.audioio import load_and_validate_track
        from seamless_loop.errors import ParameterRangeError

        info = sf.info(self.paths["short"])  # 0.5 s = 24000 frames
        with self.assertRaises(ParameterRangeError):
            load_and_validate_track(
                self.paths["short"], 0, info.frames + 100
            )


if __name__ == "__main__":
    unittest.main()
